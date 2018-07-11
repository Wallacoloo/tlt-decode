# Context

In Spring 2018, a person who goes by // formatted Eakin's [Hard Reset](https://www.fimfiction.net/story/67362/hard-reset)
book and sequels, and then crowdfunded a print run of it.
In what seems to be tradition, he left multiple easter eggs in the print.
One such easter egg is two solid pages of base64-coded data at the back of the book.

Attempts to OCR the text with off-the-shelf tools like tesseract produced poor results after
several attempts to tune the tool to the particular character set.
The font used has easily confusable glyphs for letter l and number 1,
as well as number 0 and letter O. Thankfully, it behaves much like a bitmap font, with no
ligatures.

# What this tool does
This is a Rust binary which attempts to OCR the text using a fairly straightforward method:
1. Threshold the image, and then extract the connected components (i.e. disjoint sets of connected pixels).
2. Arrange these components into discrete rows.
3. Within each row, 'merge' multiple connected components into one glyph if they have
properties that suggest they might actually be just one glyph (e.g. i and j are one glyph, despite
having two disjoint sets of connected pixels).
4. For each glyph in each row, cross-correlate it against a pre-defined mapping of
bitmaps to text. If the cross-correlation exceeds some pre-determined threshold, it's
assumed that the glyph represents the matched text.

In the event that a glyph cannot be decoded, it is printed to the console and the user
is asked to manually decode it. That bitmap is now permanently associated with the user's
response in order to decode similar glyphs.

# Usage
Everything except the bitmap -> glyph mapping is provided in this repo. Just run
```sh
cargo run --release
```
in the root directory of the repository to get started.

## Performance
This is meant to be a brute-force solution to this OCR problem. Very little
attention was given to performance. The image contains about 32000 connected components,
and each one takes multiple seconds to decode. Decoding the entire image could take
a full day.

There's some easy opportunities for performance improvement:
1. Use something like [rayon](https://crates.io/crates/rayon) to process glyphs in parallel.
2. Use SIMD when computing the cross-correlation.
3. Perform each bitmap cross-correlation (in where we consider all possible ways to
overlap the unknown glyph with a known bitmap, subtract the two, and measure the
energy in the resulting bitmap) using FFT-based 2d convolution (O(n^2 log(n))) instead
of the naive O(n^4) algorithm.

Other than that, OCR based on cross-correlating against known bitmaps seems to have
inherent performance limitations.
