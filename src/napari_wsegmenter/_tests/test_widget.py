# @pytest.mark.parametrize(
#     "shape, model_name",
#     [
#         ((100, 100), "stardist"),
#         ((64, 64), "cellpose"),
#         ((128, 128), "sam"),
#     ],
# )
# def test_segment_widget(shape, model_name):
#     im_data = np.random.random(shape)
#     thresholded = segment_widget(im_data, model_name)  # type: ignore
#     assert thresholded.shape == im_data.shape
