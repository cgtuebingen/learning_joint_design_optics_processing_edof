#!/bin/bash
# This bash script computes the results of all inference models of *_reverse_corrected*.py using the v5 inference file.
# The test iterates all models in ../train_log/*reverse_corrected*/, and generate outputs of narrow (*long*.png) and wide (*wide*.png) views. 
# Animations of these two views versus models are generated in ../temp/animation_generator.py

for iter in {2000..200000..2000}
do
	echo $iter 
	python3 psfDesign_inference_multifocal_homogeneous_blendmap_inter-view_cost_reverse_corrected_tp_0.9_v5.py --model ../train_logs/model_slm2k_2proptest_homogeneous_blendmask_inter-view_cost_reverse_corrected/model-$iter --output_wide ../temp/v5/output_wide_v5_model_$iter --output_long ../temp/v5/output_long_v5_model_$iter --output_activations ../temp/output_activations_v5 --input ~/playground/python/playground/image_blend.png 
done
#python3 psfDesign_inference_multifocal_homogeneous_blendmap_inter-view_cost_reverse_corrected_tp_0.9_v5.py --model ../train_logs/model_slm2k_2proptest_homogeneous_blendmask_inter-view_cost_reverse_corrected/model-200000 --output_wide ../temp/v5/output_wide_v5_model_200000 --output_long ../temp/v5/output_long_v5_model_200000 --output_activations ../temp/output_activations_v5 --input ~/playground/python/playground/image_blend.png 
