import cv2
import numpy as np
import os

# The final output image size
FINAL_IMAGE_WIDTH = 3000
FINAL_IMAGE_HEIGHT = 1000
# Number of orb matches needed to trust the homography
MIN_MATCHES_NEEDED = 20
# Folder for source images
QUARY_DIR = "images"

# Make a nice display
def create_named_window(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h = image.shape[0]
    w = image.shape[1]

    # Set to appropriate screen size
    WIN_MAX_SIZE = 1500
    if max(w,h) > WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE / max(w,h)
    else:
        scale = 1
    cv2.resizeWindow(winname = window_name, width = int(w * scale), height = int(h * scale))

# Meat and potatos, calculate the transform needed between frames
def calc_homography_transformation(mathces_in_subset, kp_train, kp_query):
    if len(mathces_in_subset) < MIN_MATCHES_NEEDED:
        # Not enough matches error out (moved camera too much between images)
        return None, None
    
    # Change dimenstions of the points for cv function arguments
    src_pts = np.float32([kp_train[m.trainIdx].pt for m in mathces_in_subset]).reshape(
        -1,1,2
    )
    dest_pts = np.float32([kp_query[m.queryIdx].pt for m in mathces_in_subset]).reshape(
        -1,1,2
    )

    H, _ = cv2.findHomography(srcPoints=src_pts, dstPoints=dest_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    # Return the homography
    return H

# Use ORB detection to find the matches between images
def detect_features(image, show_features=False):
    detector = cv2.ORB_create(nfeatures=3500)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_scale, mask=None)

    return keypoints, descriptors

# Seed the homography chain
def findBaseHomography():
    color_image1 = cv2.imread("{}/image01.JPG".format(QUARY_DIR))
    # For Mosaic (x,y) top left (0,0)
    source_points = np.array([[0,0],[5000,0], [0,3500], [5000,3500]])
    ortho_points = np.array([[0,0], [500,0], [0, 350], [500,350]])

    H1, _ = cv2.findHomography(srcPoints=source_points, dstPoints=ortho_points)
    return H1, color_image1

# Overlay the images
def fuse_color_images(A,B):
    # Check dimentions to make sure mask will work
    assert(A.ndim == 3 and B.ndim == 3)
    assert(A.shape == B.shape)

    # Empty array to fill
    C = np.zeros(A.shape, dtype=np.uint8)

    # Mask for each image (A, B)
    A_mask = np.sum(A, axis=2) >0
    B_mask = np.sum(B,axis=2) > 0

    # Where is there only A image
    A_only = A_mask & ~B_mask
    # Where is there only B image
    B_only = B_mask & ~A_mask
    # Where do the imags 
    A_and_B = A_mask & B_mask

    # Add in just A image 
    C[A_only] = A[A_only]
    # Add in just B image
    C[B_only] = B[B_only]
    # Add in overlay with average combinations of the two, adjust ratio if needed
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]

    # Return fused images
    return C

def main():
    # List of processed images
    warped_images = []
    # Seed the homography
    homography_previous_2_mosaic, start_image = findBaseHomography()
    # Distort the start image to final canvas
    start_warpped_image = cv2.warpPerspective(start_image, homography_previous_2_mosaic, (FINAL_IMAGE_WIDTH,FINAL_IMAGE_HEIGHT))
    # Add to list of processed images
    warped_images.append(start_warpped_image)

    # Track the index of the image name, image order is important
    prior = "01".zfill(2)
    # Set me for how many images in folder
    for i in range(2, 11):
        # Get next two images
        file_name = str(i).zfill(2)
        prior_image = cv2.imread("{}/image{}.JPG".format(QUARY_DIR, prior))
        current_image = cv2.imread("{}/image{}.JPG".format(QUARY_DIR, file_name))

        # Find the shared key points between them
        kp_train, desc_train = detect_features(current_image, show_features=False)
        kp_query, desc_query = detect_features(prior_image, show_features=False)

        match = cv2.BFMatcher.create(cv2.NORM_L2)
        matches = match.knnMatch(desc_query, desc_train, k = 2)

        # Filter only top 80% of matches
        valid = []
        for m,n in matches:
            if m.distance < 0.8 * n.distance:
                valid.append(m)

        # Calcualte the transformation mat needed to merge images to final canvas
        homography_current_2_previous = calc_homography_transformation(valid, kp_train, kp_query)
        
        # To mosaic
        # Dot product to run through transformation mat
        homography_current_2_mosaic = np.dot( homography_previous_2_mosaic, homography_current_2_previous)
        image_mosaic = cv2.warpPerspective(current_image, homography_current_2_mosaic, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))
        warped_images.append(image_mosaic)
        print("Prior {} Current {}".format(prior, file_name))
    
        # Update prior variables
        homography_previous_2_mosaic = homography_current_2_mosaic
        prior = file_name

    # Generate the final mosaic image
    final_output_image = np.zeros((FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, 3), dtype=np.uint8)
    for image_mosaic in warped_images:
        # Stich
        final_output_image = fuse_color_images(final_output_image, image_mosaic)
        # Display the images one at a time
        cv2.imshow("M", image_mosaic)
        cv2.imshow("TMP", final_output_image)
        cv2.waitKey(0)

    create_named_window("Final Image", final_output_image)
    cv2.imshow("Final Image", final_output_image)

    # Final image display    
    cv2.waitKey(0)
    cv2.imwrite("Final_personal.jpg", final_output_image)

if __name__ == "__main__":
    main()