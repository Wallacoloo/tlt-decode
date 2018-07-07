extern crate euclid;
extern crate image;
extern crate imageproc;
#[macro_use]
extern crate lazy_static;

use euclid::{
    rect,
    TypedRect
};
use image::Luma;
use imageproc::{
    contrast::threshold_mut,
    region_labelling::{connected_components, Connectivity}
};

use std::{
    collections::BTreeMap,
    fs::DirBuilder,
    path::{Path, PathBuf}
};

// hard-coded output paths
lazy_static! {
    static ref WORK_DIR: PathBuf = Path::new("intermediate").into();
    static ref SEGMENTS_DIR: PathBuf = WORK_DIR.join("segments");
}

fn label_page<P: AsRef<Path>>(file: P) {
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

    for (idx, r) in regions {
        let cropped = im.crop(r.origin.x, r.origin.y, r.size.width, r.size.height);
        cropped.save(SEGMENTS_DIR.join(format!("{:05}.png", idx)))
            .expect("failed to save debug `cropped`");
    }


    println!("done");
}

fn main() {
    // Create output directories
    for d in &[&*WORK_DIR, &*SEGMENTS_DIR] {
        DirBuilder::new()
            .recursive(true)
            .create(d)
            .expect("unable to create working directory");
    }
    label_page("merged.jpg");
}
