import cv2
import numpy as np

# Đọc ảnh lớn và template
image = cv2.imread('large_image.jpg', cv2.IMREAD_COLOR)
template = cv2.imread('template_image.jpg', cv2.IMREAD_COLOR)

# Lấy kích thước của template
template_height, template_width = template.shape[:2]

# Tìm kiếm template trong ảnh lớn
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Ngưỡng tương đồng (0.5 tương đương 50%)
threshold = 0.8
locations = np.where(result >= threshold)

# Vẽ các vị trí tìm thấy trên ảnh lớn
for point in zip(*locations[::-1]):
    top_left = point
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Đếm số lượng đối tượng tương tự
object_count = len(list(zip(*locations[::-1])))

print(f"Số lượng đối tượng tương tự: {object_count}")

# Hiển thị ảnh kết quả
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
