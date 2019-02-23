all:
	cd ./nms; python setup.py build_ext --inplace; rm -rf build; cd ../
clean:
	cd ./nms; rm *.so *.c *.cpp; cd ../
