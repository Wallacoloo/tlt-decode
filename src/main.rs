#![feature(nll)] // non-lexical lifetimes
extern crate ansi_term;
extern crate euclid;
extern crate image;
extern crate imageproc;
#[macro_use]
extern crate lazy_static;
extern crate ordered_float;
extern crate rect_iter;

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

use std::{
    collections::BTreeMap,
    fs::DirBuilder,
    io::stdin,
    ops::Deref,
    path::{Path, PathBuf}
};

// hard-coded output paths
lazy_static! {
    static ref WORK_DIR: PathBuf = Path::new("intermediate").into();
    static ref SEGMENTS_DIR: PathBuf = WORK_DIR.join("segments");
}
/// Unicode 'full block' character, i.e. a solid [black] rectangle.
const UNICODE_FULL_BLOCK: &str = "█";
/// Minimum confidence needed to classify a glyph.
const MIN_CONFIDENCE: f32 = 0.998;

#[derive(Default)]
struct GlyphClassifier {
    /// Pairs of images with the corresponding string of text they represent.
    known_glyphs: Vec<(GrayImage, String)>
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

/// Return a value between 0.0 - 1.0 indicating how similar the two images are,
/// when im is shifted by (x_shift, y_shift), and pixels where an image isn't defined
/// are defaulted to the background color (0).
fn correlate_im(ref_im: &GrayImage, im: &GrayImage, x_shift: i32, y_shift: i32) -> f32 {
    let left = 1i32 - im.width() as i32;
    let top = 1i32 - im.height() as i32;
    let right = (ref_im.width() + im.width()) as i32;
    let bot = (ref_im.height() + im.height()) as i32;
    let cum_err: i64 = RectRange::from_ranges(
        left..right,
        top..bot
    ).unwrap().iter().map(|(ref_x, ref_y)| {
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
}

impl GlyphClassifier {
    /// Make future images similar to this one decode to the given string.
    fn associate_image(&mut self, im: GrayImage, s: String) {
        self.known_glyphs.push((im, s));
    }
    fn have_user_label_image(&mut self, im: GrayImage) -> String {
        show_im(&im);
        println!("What does the above image say? (hit 'enter' if it should be ignored)");
        let mut decoded = String::new();
        stdin().read_line(&mut decoded).expect("unable to read stdin");
        decoded = decoded.trim().to_owned();
        if !decoded.is_empty() {
            self.associate_image(im, decoded.clone());
        }
        decoded
    }
    /// Match the glyph against all the previously labeled images and return
    /// the most likely match, in the form of (confidence between 0.0-1.0, decoded text).
    fn label_glyph(&self, im: &GrayImage) -> (f32, &str) {
        self.known_glyphs.iter().flat_map(|(ref_im, ref_str)| {
            // Consider each possible shift of `im` such that at least one pixel
            // overlaps `ref_im`.
            RectRange::from_ranges(
                1i32 - im.width() as i32 .. ref_im.width() as i32,
                1i32 - im.height() as i32 .. ref_im.height() as i32
            ).unwrap().iter().map(|(x_shift, y_shift)| {
                (NotNan::new(correlate_im(ref_im, im, x_shift, y_shift)).unwrap(), ref_str.deref())
            }).max()
        }).max()
        .map(|(correlation, decoded)| (correlation.into_inner(), decoded))
        .unwrap_or((0f32, ""))
    }
    fn label_page<P: AsRef<Path>>(&mut self, file: P) -> String {
        println!("loading {:?}", file.as_ref());
        let mut im = image::open(file)
            .expect("Could not open left.jpg image");

        println!("inverting/thresholding image");
        im.invert();
        let mut thresholded = im.to_luma();
        threshold_mut(&mut thresholded, 85);
        im.save(WORK_DIR.join("thresholded.png"))
            .expect("failed to save debug `im`");

        println!("segmenting image");
        let im_components = connected_components(
            &thresholded,
            Connectivity::Four,
            Luma([0u8]));

        // Assign each glyph a bounding box:
        let mut regions: BTreeMap<u32, TypedRect<u32>> = Default::default();
        for (x, y, pixel) in im_components.enumerate_pixels()
            .filter(|(_, _, px)| px.data[0] != 0)
        {
            let union_with = rect(x, y, 1, 1);
            let region = regions.entry(pixel.data[0]).or_insert(union_with);
            *region = region.union(&union_with);
        }

        let decoded_img: String = regions.iter().map(|(idx, r)| {
            let cropped = im.crop(r.origin.x, r.origin.y, r.size.width, r.size.height);
            cropped.save(SEGMENTS_DIR.join(format!("{:05}.png", idx)))
                .expect("failed to save debug `cropped`");
            let as_luma = cropped.to_luma();
            let (rank, decoded) = self.label_glyph(&as_luma);
            println!("decoding glyph gives '{}' with {} confidence.", decoded, rank);
            if rank >= MIN_CONFIDENCE {
                decoded.to_owned()
            } else {
                println!("confidence doesn't meet threshold.");
                self.have_user_label_image(as_luma)
            }
        }).collect();

        println!("done");
        decoded_img
    }
}

fn main() {
    // Create output directories
    for d in &[&*WORK_DIR, &*SEGMENTS_DIR] {
        DirBuilder::new()
            .recursive(true)
            .create(d)
            .expect("unable to create working directory");
    }
    let labeled = GlyphClassifier::default().label_page("merged.jpg");
    println!("{}", labeled);
}
