import cv2
img = cv2.imread("brain.tif")
cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def select_ROI(event, x, y, flags, param):
    global selected_ROI, resizing, resized_img, scale_factor, ROIselection_done

    print("1")

    if ROIselection_done:  # Do not allow selection after the first frame
        return

    print("2")

    # Scale coordinates back to the original resolution
    x, y = int(x / scale_factor), int(y / scale_factor)
    print("3")
    if event == cv2.EVENT_LBUTTONDOWN:  # Start selection
        selected_ROI = [x, y, x, y]
        print("4")
    
    elif event == cv2.EVENT_MOUSEMOVE and selected_ROI:  # Update rectangle dynamically
        selected_ROI[2], selected_ROI[3] = x, y
        print("5")
        # Draw the rectangle dynamically on the resized image
        temp_img = resized_img.copy()
        print("6")
        x1, y1, x2, y2 = selected_ROI
        cv2.rectangle(temp_img, (int(x1 * scale_factor), int(y1 * scale_factor)),
                      (int(x2 * scale_factor), int(y2 * scale_factor)), (255, 255, 255), 2)
        print("7")  
        cv2.imshow("Image", temp_img)
        print("8")

    elif event == cv2.EVENT_LBUTTONUP:  # Finalize selection
        if selected_ROI:
            print("9")
            selected_ROI = [min(selected_ROI[0], selected_ROI[2]), min(selected_ROI[1], selected_ROI[3]),
                             max(selected_ROI[0], selected_ROI[2]), max(selected_ROI[1], selected_ROI[3])]
        ROIselection_done = True


