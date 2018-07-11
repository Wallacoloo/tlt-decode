#![feature(nll)] // non-lexical lifetimes
extern crate ansi_term;
extern crate base64;
extern crate euclid;
extern crate image;
extern crate imageproc;
#[macro_use]
extern crate lazy_static;
extern crate ordered_float;
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
    region_labelling::{connected_components, Connectivity}
};
use ordered_float::NotNan;
use rect_iter::RectRange;
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
const MIN_CONFIDENCE: f32 = 0.998;

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

/// Return a value between 0.0 - 1.0 indicating how similar the two images can
/// be made when considering all possible (x_shift, y_shift) shifts, and pixels
/// where an image isn't defined are defaulted to the background color (0).
fn correlate_im(ref_im: &GrayImage, im: &GrayImage) -> f32 {
    let left = 1i32 - im.width() as i32;
    let top = 1i32 - im.height() as i32;
    let right = (ref_im.width() + im.width()) as i32;
    let bot = (ref_im.height() + im.height()) as i32;
    let all_coordinates = RectRange::from_ranges(
        left..right,
        top..bot
    ).unwrap();
    let valid_shifts = RectRange::from_ranges(
        left .. ref_im.width() as i32,
        top .. ref_im.height() as i32
    ).unwrap();

    let correlated_once = |x_shift: i32, y_shift: i32| -> f32 {
        let cum_err: i64 = all_coordinates.iter().map(|(ref_x, ref_y)| {
            // Get the pixel at the relevant coordinates, or (0) if OOB.
            let ref_px = if ref_x >= 0 && ref_x < ref_im.width() as i32
                && ref_y >= 0 && ref_y < ref_im.height() as i32
            {
                ref_im.get_pixel(ref_x as u32, ref_y as u32).data[0]
            } else { 0 } as i32;

            let im_x = ref_x + x_shift;
            let im_y = ref_y + y_shift;
            let im_px = if im_x >= 0 && im_x < im.width() as i32
                && im_y >= 0 && im_y < im.height() as i32
            {
                im.get_pixel(im_x as u32, im_y as u32).data[0]
            } else { 0 } as i32;
            // return square error.
            ((ref_px - im_px) * (ref_px - im_px)) as i64
        }).sum();
        let max_err = 255*255 * ((right-left) * (bot-top)) as i64;
        1.0f32 - cum_err as f32 / max_err as f32
    };
    valid_shifts.into_iter().map(|(x_shift, y_shift)| {
            NotNan::new(correlated_once(x_shift, y_shift)).unwrap()
        }).max()
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
        self.known_glyphs.iter()
            .filter_map(|(ref_im, ref_str)| {
                let cor = correlate_im(ref_im, im);
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
            println!("OCR'd a line: {}", decoded_line);
            parsed_rows.push(decoded_line);
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
