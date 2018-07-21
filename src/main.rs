#![feature(nll)] // non-lexical lifetimes
extern crate ansi_term;
extern crate base64;
extern crate euclid;
extern crate image;
extern crate imageproc;
#[macro_use]
extern crate lazy_static;
extern crate ordered_float;
extern crate rayon;
extern crate rect_iter;
extern crate sha2;

use ansi_term::{
    ANSIStrings,
    Colour
};
use euclid::{
    rect,
    TypedRect
};
use image::{
    GenericImage,
    GrayImage,
    Luma
};
use imageproc::{
    contrast::threshold_mut,
    region_labelling::{connected_components, Connectivity},
    template_matching::{match_template, MatchTemplateMethod}
};
use ordered_float::NotNan;
use rayon::prelude::*;
use sha2::{Sha256, Digest};

use std::{
    collections::BTreeMap,
    fs::{DirBuilder, File, read_dir},
    io::{stdin, Write},
    ops::Deref,
    path::{Path, PathBuf}
};

// hard-coded output paths
lazy_static! {
    static ref WORK_DIR: PathBuf = Path::new("intermediate").into();
    static ref SEGMENTS_DIR: PathBuf = WORK_DIR.join("segments");
    static ref KNOWN_GLYPHS_DIR: PathBuf = WORK_DIR.join("known_glyphs");
}
/// Unicode 'full block' character, i.e. a solid [black] rectangle.
const UNICODE_FULL_BLOCK: &str = "█";
/// Minimum confidence needed to classify a glyph.
const MIN_CONFIDENCE: f32 = 0.99;

/// Takes an image and OCRs the text inside it.
/// Assumes the font is roughly a bitmap font and letters are arranged clearly into rows.
struct GlyphClassifier {
    /// Pairs of images with the corresponding string of text they represent.
    known_glyphs: Vec<(GrayImage, String)>
}

/// Contains each glyph region within one line of text.
struct Row {
    /// Glyph regions, ordered by the x coordinate of their left edge.
    regions: Vec<TypedRect<u32>>,
    // Coordinates which inclusively bound all the regions.
    // These _could_ be calculated on-the-fly, but that actually makes things messier.
    top: u32,
    bot: u32,
}

/// Return whether two regions overlap if they were extended infinitely on the vertical axis.
fn is_x_overlap(r0: &TypedRect<u32>, r1: &TypedRect<u32>) -> bool {
    // if r0 is wholly to the left or wholly to the right of r1, then no intersection,
    // else intersection.
    !(r0.max_x() < r1.min_x() || r0.min_x() > r1.max_x())
}
/// Return whether two regions overlap if they were extended infinitely on the horizontal axis.
fn is_y_overlap(r0: &TypedRect<u32>, r1: &TypedRect<u32>) -> bool {
    // if r0 is wholly above or wholly below r1, then no intersection,
    // else intersection
    !(r0.max_y() < r1.min_y() || r0.min_y() > r1.max_y())
}

/// Display the image to the console via ansi control codes/text.
fn show_im(im: &GrayImage) {
    for y in 0..im.height() {
        let row: Vec<_> = (0..im.width()).map(|x| {
            let lum = 255 - im.get_pixel(x, y).data[0];
            Colour::RGB(lum, lum, lum).paint(UNICODE_FULL_BLOCK)
        }).collect();
        println!("{}", ANSIStrings(&row));
    }
}

/// Return the maximum cross-correlation between `im` and `ref_im`, where
/// 1.0 indicates identical images, and 0.0 indicates no similarity.
///
/// i.e. consider all pairs of (`im_shift`, `ref_im`), where `im_shift` is
/// `im` but shifted by some vector such that the two images still overlap.
/// For each pair:
///   1. Perform element-wise subtraction of the two images (pixels which don't
///      exist in one images are defaulted to background color (0).
///   2. Measure the energy in the resulting image.
/// If the energy (sum of squares of each pixel value) is low, that indicates
/// the images are similar.
///
/// Therefore, return the minimum energy found, subtracted from the maximum energy
/// by which two images _could_ differ, and normalized such that 1.0 indicates
/// 100% matching images.
fn cross_correlate_im(ref_im: &GrayImage, im: &GrayImage) -> f32 {
    // Prepare correlation so that we never have to look at pixels that would be OOB.
    // imageproc will place the template only in places where it fits entirely inside
    // the other image.
    // Step 1: create an expanded_ref im which is `ref_im` surrounded by so much
    // empty space. Placing the origin of `ref_im` anywhere inside `im` should
    // be such that `expanded_ref` covers every pixel of `im`.
    let ref_pad_x = im.width()-1;
    let ref_pad_y = im.height()-1;
    let mut expanded_ref = GrayImage::new(
        ref_im.width() + 2*ref_pad_x,
        ref_im.height() + 2*ref_pad_y);
    expanded_ref.copy_from(ref_im, ref_pad_x, ref_pad_y);
    // Step 2: Expand `im` into `expanded_im`.
    // Add ref_im.width()-1 + ref_pad_x to the left edge:
    // this allows for the left-most overlap to contain just the right-most column of the
    // original ref_im.
    // Add same amount to right edge, such that the right-most overlap contains
    // just the left-most column of ref_im.
    let im_pad_x = ref_im.width()-1 + ref_pad_x;
    let im_pad_y = ref_im.height()-1 + ref_pad_y;
    let mut expanded_im = GrayImage::new(
        im.width() + 2*im_pad_x,
        im.height() + 2*im_pad_y);
    expanded_im.copy_from(im, im_pad_x, im_pad_y);
    // Correlate the template against the image at each possible pixel shift.
    let corr = match_template(
        &expanded_im,
        &expanded_ref,
        MatchTemplateMethod::SumOfSquaredErrors);
    // Take the shift that produced the best correlation and turn that into a confidence.
    let cum_err = corr.enumerate_pixels()
        .map(|(_x, _y, p)| p.data[0])
        .flat_map(NotNan::new)
        .min()
        .map(NotNan::into_inner)
        .unwrap_or(std::f32::INFINITY);
    // Maximum possible error: all pixels are completely different.
    let max_err = 255*255 * (im.width()*im.height() + ref_im.width()*ref_im.height()) as u64;
    1.0f32 - cum_err / max_err as f32
}

/// Given an image (one which has already been thresholded), find all disjoint sets
/// of connected pixels and return bounding boxes around each set of connected pixels.
fn extract_regions(thresholded: &GrayImage) -> Vec<TypedRect<u32>> {
    let im_components = connected_components(
        thresholded,
        Connectivity::Four,
        Luma([0u8]));

    let mut regions: BTreeMap<u32, TypedRect<u32>> = Default::default();
    for (x, y, pixel) in im_components.enumerate_pixels()
        .filter(|(_, _, px)| px.data[0] != 0)
    {
        let union_with = rect(x, y, 1, 1);
        let region = regions.entry(pixel.data[0]).or_insert(union_with);
        *region = region.union(&union_with);
    }
    regions.values().cloned().collect()
}

/// Group regions into discrete rows.
/// Within each row, the glyphs are ordered by their left edge.
fn arrange_regions_into_rows(regions: Vec<TypedRect<u32>>) -> Vec<Row> {
    // Assign each region to a row:
    let mut rows: Vec<Row> = Default::default();
    for rect in regions {
        match rows.iter_mut().filter(|row|
            row.intersects(&rect)
        ).next()
        {
            Some(row) => row.push(rect),
            None => rows.push(Row::from_region(rect)),
        }
    }
    // sort the regions within each row.
    rows.iter_mut().for_each(Row::sort);
    // merge certain regions, where we expect they are actually one glyph (e.g. i or j).
    for row in rows.iter_mut() {
        let mut current_region: Option<TypedRect<u32>> = None;
        let mut new_row: Vec<TypedRect<u32>> = row.regions.drain(..).flat_map(|region| {
                if let Some(r0) = current_region.take() {
                    if is_x_overlap(&r0, &region) && !is_y_overlap(&r0, &region) {
                        // the two regions sit on top of eachother; likely the two halves
                        // of something like an 'i' or 'j'.
                        current_region = Some(r0.union(&region));
                        None
                    } else {
                        // regions are distinct; yield the previous one.
                        current_region = Some(region);
                        Some(r0)
                    }
                } else {
                    // first region in sequence; save it away for processing.
                    current_region = Some(region);
                    None
                }
            })
            .collect();
        if let Some(r) = current_region {
            new_row.push(r);
        }
        row.regions = new_row;
    }
    rows
}

impl GlyphClassifier {
    fn new() -> Self {
        let images = read_dir(&*KNOWN_GLYPHS_DIR)
            .expect("error reading glyph directory")
            // unwrap Iter<Item=Result<DirEntry>> to Iter<Item=DirEntry>
            .map(|dir_entry| dir_entry.expect("error reading glyph directory"))
            // map Iter<Item=DirEntry> to Iter<Item=(GrayImage, String)>
            .map(|dir_entry| {
                // Parse glyph_name from "<glyph_name>.<sha>.png".
                let glyph_name = dir_entry.file_name()
                    .into_string()
                    .unwrap()
                    .split('.')
                    .next()
                    .unwrap()
                    .replace("_", "/");
                let image = image::open(dir_entry.path())
                    .expect("Unable to open glyph")
                    .to_luma();
                (image, glyph_name)
            });
        Self {
            known_glyphs: images.collect()
        }
    }
    /// Make future images similar to this one decode to the given string.
    fn associate_image(&mut self, im: GrayImage, s: String) {
        let mut hasher = Sha256::default();
        hasher.input(&im);
        let hash = base64::encode_config(&hasher.result(), base64::URL_SAFE);
        // Saving a file with name of 'slash' causes issues; represent that character with '_'.
        let name = KNOWN_GLYPHS_DIR.join(format!("{}.{}.png", s.replace("/", "_"), hash));
        println!("Saving new image association to {:?}", name);
        im.save(name).expect("failed to save glyph association");
        self.known_glyphs.push((im, s));
    }
    /// Show an image to the console, ask the user what symbol it corresponds to,
    /// and then associate that glyph with that symbol permanently.
    fn have_user_label_image(&mut self, im: GrayImage) -> String {
        show_im(&im);
        println!("What does the above image say? (hit 'enter' if it should be ignored)");
        let mut decoded = String::new();
        stdin().read_line(&mut decoded).expect("unable to read stdin");
        decoded = decoded.trim().to_owned();
        self.associate_image(im, decoded.clone());
        decoded
    }
    /// Match the glyph against all the previously labeled images and return
    /// the most likely match, in the form of (confidence between 0.0-1.0, decoded text).
    fn label_glyph(&self, im: &GrayImage) -> (f32, &str) {
        self.known_glyphs.par_iter()
            .filter_map(|(ref_im, ref_str)| {
                let cor = cross_correlate_im(ref_im, im);
                if !ref_str.is_empty() || cor == 1.0f32 {
                    // Only decode to empty glyph if 100% match.
                    // TODO: why do so many things decode to empty glyphs
                    // if this is omitted?
                    Some((NotNan::new(cor).unwrap(), ref_str.deref()))
                } else {
                    None
                }
            })
            .max()
            .map(|(correlation, decoded)| (correlation.into_inner(), decoded))
            .unwrap_or((0f32, ""))
    }
    /// Take an image of multiple lines of text and OCR the whole thing.
    fn label_page<P: AsRef<Path>>(&mut self, file: P) -> String {
        println!("loading {:?}", file.as_ref());
        let mut im = image::open(file)
            .expect("Could not open input image");

        println!("inverting/thresholding image");
        im.invert();
        let mut thresholded = im.to_luma();
        threshold_mut(&mut thresholded, 85);
        im.save(WORK_DIR.join("thresholded.png"))
            .expect("failed to save debug `im`");

        let regions = extract_regions(&thresholded);

        let rows = arrange_regions_into_rows(regions);

        for r in &rows {
            println!("row from {} to {} contains {} items", r.top, r.bot, r.regions.len());
        }
        println!("total items to parse: {}",
            rows.iter().map(|r| r.regions.len()).sum::<usize>()
        );

        let mut b64_file = File::create(WORK_DIR.join("b64.txt"))
            .expect("failed to create output base64 text file");

        // Perform the actual letter-by-letter OCR.

        let mut parsed_rows = Vec::new();
        for (row_idx, row) in rows.into_iter().enumerate() {
            let decoded_line: String = row.regions.into_iter().enumerate().map(|(col_idx, region)| {
                let r = region;
                let cropped = im.crop(r.origin.x, r.origin.y, r.size.width, r.size.height);
                let as_luma = cropped.to_luma();
                let (rank, decoded) = self.label_glyph(&as_luma);
                println!("decoding glyph gives '{}' with {} confidence.", decoded, rank);
                let decoded = if rank >= MIN_CONFIDENCE {
                    decoded.to_owned()
                } else {
                    println!("confidence doesn't meet threshold.");
                    self.have_user_label_image(as_luma)
                };
                cropped.save(SEGMENTS_DIR.join(format!("{:03}-{:03}-{}.png", row_idx, col_idx, decoded.replace("/", "_"))))
                    .expect("failed to save debug `cropped`");
                b64_file.write(decoded.as_bytes())
                    .expect("failed to write OCR'd text to output file");
                decoded
            }).collect();
            if !decoded_line.is_empty() {
                println!("OCR'd a line: {}", decoded_line);
                b64_file.write("\n".as_bytes())
                    .expect("failed to write newline to output file");
                parsed_rows.push(decoded_line);
            }
        }

        println!("done");
        parsed_rows.into_iter().collect()
    }
}

impl Row {
    /// Create a row which contains exactly one glyph region.
    fn from_region(region: TypedRect<u32>) -> Self {
        Self {
            regions: vec![region],
            top: region.min_y(),
            bot: region.max_y()
        }
    }
    /// Returns true if the region intersects with the bounding box of this row.
    fn intersects(&self, region: &TypedRect<u32>) -> bool {
        // Check if region is entirely above or entirely below the row;
        // invert that to determine if there's intersection.
        !(region.max_y() < self.top || region.min_y() > self.bot)
    }
    /// Add a new region to the row.
    fn push(&mut self, region: TypedRect<u32>) {
        self.top = self.top.min(region.min_y());
        self.bot = self.bot.max(region.max_y());
        self.regions.push(region);
    }
    fn sort(&mut self) {
        self.regions.sort_by_key(|r| {
            (r.min_x(), r.max_x(), r.min_y(), r.max_y())
        });
    }
}


fn main() {
    // Create output directories
    for d in &[&*WORK_DIR, &*SEGMENTS_DIR, &*KNOWN_GLYPHS_DIR] {
        DirBuilder::new()
            .recursive(true)
            .create(d)
            .expect("unable to create working directory");
    }
    let labeled = GlyphClassifier::new().label_page("merged.jpg");
    println!("{}", labeled);
}
