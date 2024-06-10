import cv2

# Read the image
image = cv2.imread("marker_5_1_55_crop.png")

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Canny edge detection
# edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# # Combine edges with original image
# edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# result = cv2.bitwise_and(image, edges)
# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
# cv2.resizeWindow("Image", 800, 600)
# # Display the result
# cv2.imshow("Edge Detection Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# Create a named window with a specific size
cv2.namedWindow("Edge Detection Result", cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
cv2.resizeWindow("Edge Detection Result", 800, 600)  # Set the size of the window (width, height)

# Display the edge-detected image in the window
cv2.imshow("Edge Detection Result", edges)

# Wait for a key press and close the window when a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
