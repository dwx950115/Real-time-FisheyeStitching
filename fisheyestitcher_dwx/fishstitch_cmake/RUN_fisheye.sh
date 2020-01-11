# Change these numbers
#========================================================================
# Directory setup
output_dir_root='./result_video'
intersectionId=1
positionId=1
#========================================================================

BUILD_DIR='./build'
# output dir
output_dir="${output_dir_root}/${intersectionId}"

# Build
if [ ! -d "$BUILD_DIR" ]; then
    mkdir $BUILD_DIR

    cd $BUILD_DIR
    cmake ..
    make -j 4
    cd ..
fi
# Update
if [ -d "$BUILD_DIR" ]; then
    cd $BUILD_DIR
    make -j 4
    cd ..
fi

#---- stitch frame1 and frame2
echo "-------------------------------------------------"
echo "Stitching dual fisheye image: "
echo "-------------------------------------------------"
./build/fisheye_capture --dir $output_dir\
                        --intersectionId $intersectionId\
                        --positionId $positionId  
