use opencv::{
    core,
    highgui,
    imgproc,
    prelude::*,
    Result,
    imgcodecs,
};

use ndarray::Array2;
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::imgproc::COLOR_BGR2GRAY;


fn main() {
    let img = imgcodecs::imread("", IMREAD_COLOR).unwrap();
    highgui::imshow("image", &img).unwrap();
    highgui::wait_key(0).unwrap();

    let mut gray_img_= core::Mat::default();
    imgproc::cvt_color(&img,  &mut gray_img_ ,COLOR_BGR2GRAY,0).unwrap();

    let mut thresh_img = core::Mat::default();
    //imgproc::threshold(&gray_img_, &mut thresh_img, 0.0, 255.0, imgproc::THRESH_BINARY_INV | imgproc::THRESH_OTSU).unwrap();
    imgproc::threshold(&gray_img_, &mut thresh_img, 130.0, 255.0, imgproc::THRESH_BINARY_INV).unwrap();

    highgui::imshow("image", &thresh_img).unwrap();
    highgui::wait_key(0).unwrap();

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(3, 3),
        core::Point::new(-1, -1),
    ).unwrap();

    let mut open_img = core::Mat::default();
    imgproc::morphology_ex(&thresh_img, &mut open_img, imgproc::MORPH_OPEN, &kernel, core::Point::default(), 1, core::BORDER_CONSTANT, core::Scalar::default()).unwrap();

    let mut sure_bg = core::Mat::default();
    imgproc::dilate(&open_img, &mut sure_bg, &kernel, core::Point::default(), 3, core::BORDER_CONSTANT, core::Scalar::default()).unwrap();

    highgui::imshow("image", &sure_bg).unwrap();
    highgui::wait_key(0).unwrap();

    let mut dist_transform = core::Mat::default();
    imgproc::distance_transform(&open_img, &mut dist_transform, imgproc::DIST_L2, 5, core::CV_32F).unwrap();

    // Find the maximum value in the distance transform
    let mut min_val = 0f64;
    let mut max_val = 0f64;
    core::min_max_loc(&dist_transform, Some(&mut min_val), Some(&mut max_val), None, None, &core::no_array()).unwrap();

    // Apply threshold at 50% of the maximum value
    let mut sure_fg = core::Mat::default();
    imgproc::threshold(&dist_transform, &mut sure_fg, 0.5 * max_val, 255.0, imgproc::THRESH_BINARY).unwrap();

    highgui::imshow("image", &sure_fg).unwrap();
    highgui::wait_key(0).unwrap();

    let mut sure_fg_u8 = core::Mat::default();
    sure_fg.convert_to(&mut sure_fg_u8, core::CV_8U, 1.0, 0.0).unwrap();

    let mut unknown = core::Mat::default();
    core::subtract(&sure_bg, &sure_fg_u8, &mut unknown, &core::no_array(), core::CV_8U).unwrap();

    highgui::imshow("image", &unknown).unwrap();
    highgui::wait_key(0).unwrap();

    let mut markers = core::Mat::default();
    imgproc::connected_components(&sure_fg_u8, &mut markers, 8, core::CV_32S).unwrap();

    // Increment all labels in markers by 1
    let mut incremented_markers = core::Mat::default();
    core::add(&markers, &core::Scalar::all(1.0), &mut incremented_markers, &core::no_array(), -1).unwrap();
    markers = incremented_markers;

    // Set the marker for unknown regions to 0
    for i in 0..markers.rows() {
        for j in 0..markers.cols() {
            if *unknown.at_2d::<u8>(i, j).unwrap() == 255 {
                *markers.at_2d_mut::<i32>(i, j).unwrap() = 0;
            }
        }
    }
    imgproc::watershed(&img, &mut markers).unwrap();

    let mut markers_normalized = core::Mat::default();
    core::normalize(
        &markers, &mut markers_normalized, 
        0.0, 255.0, 
        core::NORM_MINMAX, 
        core::CV_8UC1, 
        &core::no_array()
    ).unwrap();


    highgui::imshow("image", &markers_normalized).unwrap();
    highgui::wait_key(0).unwrap();

}
