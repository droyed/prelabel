.PHONY: help test demo clean

help:
	@echo "Available targets:"
	@echo "  test   - Run test_LabelStudioClient.py"
	@echo "  demo   - Run demo_LabelStudioClient.py"
	@echo "  clean  - Remove generated ZIP files and Python bytecode cache"

test:
	python tests/test_LabelStudioClient.py
	python tests/test_adapters.py

demo:
	python demos/demo_LabelStudioClient.py
	python demos/demo_get_started.py

clean:
	rm -f test_yolo.zip test_coco.zip test_png.zip
	rm -f smart_city_yolo.zip medical_seg_coco.zip
	rm -f yolo_bboxes.zip coco_polygons.zip png_masks.zip
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
