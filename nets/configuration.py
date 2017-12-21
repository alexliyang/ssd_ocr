image_size = 300.0
#oh
layer_boxes = [12, 12, 12, 12, 12, 12]
classes = 1

box_ratios = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

#[0.2, 0.36, 0.52, 0.68, 0.84, 1.0]
scales = [0.2 + i*0.8/5  for i in range(6)],

box_s_min = 0.1
negposratio = 3

# to be set programmatically
out_shapes = None
defaults = None
