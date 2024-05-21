from segment_anything import build_sam, SamAutomaticMaskGenerator
import cv2
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="checkpoints/sam_vit_h_4b8939.pth"))
image = cv2.imread("jpeg/frame_75.jpg")
#print("image", image)
masks = mask_generator.generate(image)
print("masks", masks[0].keys())
