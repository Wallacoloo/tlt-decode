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
    GrayImage,
    Luma
};
use imageproc::{
    contrast::threshold_mut,
    region_labelling::{connected_components, Connectivity},
};
use ordered_float::NotNan;
use rect_iter::RectRange;
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
const MIN_CONFIDENCE: f32 = 0.986;
/// Minimum value a pixel needs to be considered foreground.
const FG_THRESHOLD: u8 = 85;

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
/// Returns whether `inside` is fully contained within `superset` if superset is extended
/// infinitely on the vertical axis.
fn is_x_fully_contained(inside: &TypedRect<u32>, superset: &TypedRect<u32>) -> bool {
    inside.min_x() >= superset.min_x() && inside.max_x() <= superset.max_x()
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

/// Return the maximum cross-correlation between `im` and `pat`, where
/// 1.0 indicates identical images, and 0.0 indicates no similarity.
///
/// i.e. consider all pairs of (`im`, `pat_shift`), where `pat_shift` is
/// `pat` but shifted by some vector such that the two images still overlap.
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
fn cross_correlate_im(im: &GrayImage, pat: &GrayImage) -> f32 {
    const MAX_SHIFT_X: i32 = 4;
    const MAX_SHIFT_Y: i32 = 5;
    let valid_shifts = RectRange::from_ranges(
        -MAX_SHIFT_X..MAX_SHIFT_X+1,
        -MAX_SHIFT_Y..MAX_SHIFT_Y+1
    ).unwrap();

    let sum_of_all_squares: u64 = im.pixels()
        .chain(pat.pixels())
        .map(|v|
            v.data[0] as u64 * v.data[0] as u64
        ).sum::<u64>();
    let max_err = 255*255 * (im.width()*im.height() + pat.width()*pat.height()) as u64;

    let correlated_once = |(x_shift, y_shift) : (i32, i32)| -> f32 {
        let im_range = RectRange::zero_start(im.width() as i32, im.height() as i32).unwrap();
        let pat_range = RectRange::zero_start(pat.width() as i32, pat.height() as i32).unwrap();
        let overlapping = im_range.intersection(
            &pat_range.slide((x_shift, y_shift))
        );
        let inner_product = overlapping.into_iter()
            .flatten()
            .map(|(x, y)| {
                let im_px = im.get_pixel(x as u32, y as u32).data[0];
                let pat_px = pat.get_pixel((x - x_shift) as u32, (y - y_shift) as u32).data[0];
                // Subtract this pixel's contribution to sum_of_all_squares:
                // - im_px^2 - pat_px^2
                // Add the square error:
                // + (im_px - pat_px)^2
                // This comes out to -2*im_px*pat_px
                // Factor out the -2
                im_px as u64 * pat_px as u64
            }).sum::<u64>();
        let cum_err = sum_of_all_squares - 2*inner_product;
        (max_err - cum_err) as f32 / max_err as f32
    };
    valid_shifts.into_iter()
        .map(correlated_once)
        .flat_map(NotNan::new)
        .max()
        .map(NotNan::into_inner)
        .unwrap_or(0f32)
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
                    if is_x_overlap(&r0, &region) && !is_y_overlap(&r0, &region)
                        || is_x_fully_contained(&region, &r0) {
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
        threshold_mut(&mut thresholded, FG_THRESHOLD);
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
                    println!("confidence doesn't meet threshold for glyph at row {}, col {}", row_idx, col_idx);
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
