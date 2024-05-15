from skimage import transform
from skimage.feature import match_descriptors, ORB, SIFT
from skimage.measure import ransac

def rectify(ref_image, slave_image):
    '''
        This method is responsible for aligning the slave image on the reference image using Sift Detector
        This is done by calling the detect and extract on both images, returning the keypoints, and the descriptors 
        then mathcing these descrpitors together
    '''
    sift = SIFT()

    sift.detect_and_extract(ref_image)
    keypoints1, descriptors1 = sift.keypoints, sift.descriptors
    sift.detect_and_extract(slave_image)
    keypoints2, descriptors2 = sift.keypoints, sift.descriptors
        
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
    src, dst = keypoints1[matches[:, 0]], keypoints2[matches[:, 1]]
    
    # tform = transform.AffineTransform()
    # tform.estimate(src, dst)
    
    '''
        It robustly estimate transform model with RANSAC (Random Sample Consensus)
        It is a robust algorithm used for estimating the parameters of a mathematical model 
        from a set of observed data that contains outliers
        The Consensus role is to find the model which fits the largest subset of our data points.
    '''
    tform, _ = ransac(
        (src, dst), transform.ProjectiveTransform, min_samples=3, residual_threshold=2, max_trials=100
    )
    
    # This applies transformatoin to the slave image to align it to the reference image.
    rectified_image = transform.warp(slave_image, tform)
    return rectified_image
    
def main():
    from skimage import io
    import matplotlib.pyplot as plt    
        
    image1 = io.imread('data/A/0055.png', as_gray=True)
    image2 = io.imread('data/B/0055.png', as_gray=True)

    rectified_image = rectify(image1, image2)

    # Plot original images and registered image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image1)
    plt.title('Image 1')

    plt.subplot(1, 3, 2)
    plt.imshow(image2)
    plt.title('Image 2')

    plt.subplot(1, 3, 3)
    plt.imshow(rectified_image)
    plt.title('Registered Image')

    plt.show()
