# Image Recognition 
> by Yuan, Yiming Harry

## imagenet


### Example 1
![](0.gif)
```
henry@henry-ThinkPad-T430:~/PycharmProjects/ImageRecognition/imagenet$ python classify_image.py --image_file 0.jpg
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/framework/op_def_util.cc:332] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
aircraft carrier, carrier, flattop, attack aircraft carrier (score = 0.92322)
warplane, military plane (score = 0.00970)
liner, ocean liner (score = 0.00770)
drilling platform, offshore rig (score = 0.00564)
trimaran (score = 0.00369)
```


### Example 2
![](2.gif)
```
henry@henry-ThinkPad-T430:~/PycharmProjects/ImageRecognition/imagenet$ python classify_image.py --image_file 2.jpg
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/framework/op_def_util.cc:332] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
cab, hack, taxi, taxicab (score = 0.40581)
traffic light, traffic signal, stoplight (score = 0.23709)
obelisk (score = 0.03999)
fountain (score = 0.03717)
suspension bridge (score = 0.01233)
```

### Reference
[1]. How to Build a Simple Image Recognition System with TensorFlow (http://www.wolfib.com/Image-Recognition-Intro-Part-1/)
