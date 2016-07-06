g++ src/*.cpp -pthread -O2 -o bin/fdcnn_train -Wall
g++ src/*.cpp -pthread -O2 -o bin/fdcnn_predict -Wall -D FDCNN_PREDICT
g++ src/*.cpp -pthread -O2 -o bin/fdcnn_getfeat -Wall -D FDCNN_GETFEAT
g++ src/*.cpp -pthread -O2 -o bin/fdcnn_switch_model -Wall -D FDCNN_SWITCH_MODEL
