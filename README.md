# jetson_multimedia_api_examples

### syncSensorSerhiy
This example could be used for dumping gray imag of two CSI cameras connected to Jetson Nano


### How to build

- To build, copy example folder to corresponding folder in `/usr/src/jetson_multimedia_api/`. For example, `argus/samples/syncSensorSerhiy` should be copied to `/usr/src/jetson_multimedia_api/argus/samples/syncSensorSerhiy`
- create build folder in `/usr/src/jetson_multimedia_api/argus/`
- edit `/usr/src/jetson_multimedia_api/argus/CMakeLists.txt` file and insert `add_subdirectory(samples/syncSensorSerhiy)` under `add_subdirectory(samples/syncSensor)`
- Then:
```
cd build
cmake ../
cd samples/syncSensorSerhiy
make
```
