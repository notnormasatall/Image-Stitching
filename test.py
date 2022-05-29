from image_stitcher import Interface

if __name__ == "__main__":
    stitcher = Interface()
    stitcher.process_option("/upload," +
                            "images/sample1.2.JPG," +
                            "images/sample1.1.JPG"
                            )
    stitcher.process_option("/showres,images/res/sample1res.JPG")
    stitcher.process_option("/clear")
    #
    #
    stitcher.process_option("/upload," +
                            "images/sample2.4.JPG," +
                            "images/sample2.3.JPG," +
                            "images/sample2.2.JPG," +
                            "images/sample2.1.JPG"
                            )
    stitcher.process_option("/showres,images/res/sample2res.JPG")
    stitcher.process_option("/clear")
    #
    #
    stitcher.process_option("/upload," +
                            "images/sample3.2.JPG," +
                            "images/sample3.1.JPG"
                            )
    stitcher.process_option("/showres,images/res/sample3res.JPG")
    stitcher.process_option("/clear")


    stitcher.process_option("/upload," +
                            "images/sample4.1.JPG," +
                            "images/sample4.2.JPG"
                            )
    stitcher.process_option("/drawmatches,1,2")
    stitcher.process_option("/showres,images/res/sample4res.JPG")
    stitcher.process_option("/clear")
    stitcher.process_option("/exit")
