The examples provided here were extracted from the Brats 2020 dataset. Slices were taken from subject 2 of the training set and the matrix was transposed to obtain the current view (important step). Additionally, the examples from the test set underwent the entire processing pipeline.

When generating the synthetic image, the script removes the black borders as they can affect the synthesis. If you try using different images, you will notice that the resulting size is 128 x 128, with the black borders reduced. This is important step since the black margins affects to the synthesis

It is strongly recommended to apply HD-BET before synthesis in order to remove the skull. If a different procedure is used, the resulting outcome may not be guaranteed.

Finally, as the focus is on obtaining synthetic T2 images, the generation.py script specifically converts T1 images to T2.
