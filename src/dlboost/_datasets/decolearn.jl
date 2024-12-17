# using FilePathsBase; using FilePathsBase: /
using FilePaths;
using FilePaths: /;
using Glob, HDF5
using Combinatorics: combinations
using Base.Iterators: product, flatten
using BenchmarkTools
using PythonCall

function change_array_layout(arr::AbstractArray)
	# change array column major to row major
	# for python program to call
	permutedims(arr, reverse(1:ndims(arr)))
end

function get_data_from_h5(file_path::String, ch::Int)
	# for python program to call
	# when juliacall update, change this code
	# phase = phase+1
	# contrast = contrast+1
	ch = copy(ch) + 1
	d = h5open(file_path, "r") do fid
		Dict{String, Array{T} where T}(
			# "cse" => change_array_layout(
			#     fid["cse_real"][:,:,:,ch]+1im*fid["cse_imag"][:,:,:,ch]),
			# "image_1ch" => change_array_layout(
			#     fid["img_real"][:,:,:]+1im*fid["img_imag"][:,:,:]),
			"kspace_data" => change_array_layout(
				fid["kspace_data_z_real"][:, :, :, ch] + 1im * fid["kspace_data_z_imag"][:, :, :, ch]),
			"kspace_density_compensation" => change_array_layout(
				fid["kspace_density_compensation"][:, :]),
			"kspace_traj" => change_array_layout(
				fid["kspace_traj_real"][:, :] + 1im * fid["kspace_traj_imag"][:, :]),
			# "image" => change_array_layout(
			# 	fid["multi_ch_img_real"][:, :, :, ch] + 1im * fid["multi_ch_img_imag"][:, :, :, ch]),
			# "kspace_mask" => change_array_layout(
			# 	fid["kspace_data_mask"][:, :, :, ch]),
			# "recon_update" => change_array_layout(
			#     fid["recon_update_real"][:,:,:,phase,contrast]+1im*fid["recon_update_imag"][:,:,:,phase,contrast])
		)
	end
	return d
end
# subjects = @benchmark get_data_from_h5("/data-local/anlab/Chunxu/DL_MOTIF/recon_results/CCIR_01168_ONC-DCE/ONC-DCE-004/contrast_0_phase_0.h5", 1)

function get_data_from_h5(file_path::String)
	# for python program to call
    # we can't summation different channel, because kspace have phase shift, lead to data lost.
	d = h5open(file_path, "r") do fid
		Dict{String, Array{T} where T}(
			"kspace_data" => change_array_layout(
				fid["kspace_data_z_real"][:,:,:,:] + 1im *fid["kspace_data_z_imag"][:,:,:,:]),
			"kspace_density_compensation" => change_array_layout(
				fid["kspace_density_compensation"][:,:]),
			"kspace_traj" => change_array_layout(
				fid["kspace_traj_real"][:,:] + 1im * fid["kspace_traj_imag"][:,:])
		)
	end
	return d
end
# subjects = @benchmark get_data_from_h5("/data-local/anlab/Chunxu/DL_MOTIF/recon_results/CCIR_01168_ONC-DCE/ONC-DCE-004/contrast_0_phase_0.h5", 1)

function check_top_k_channel(file_path::String, k::Int = 5)
	top_channels = h5open(file_path, "r") do fid
		len,sp,kz,ch= size(fid["kspace_data_z_real"])
		center_len =  Int(round(len/2))
		center_z =  Int(round(kz/2))
		lowk_energy = zeros(ch)  
		for ch in 1:ch
			lowk_energy[ch] = abs.(fid["kspace_data_z_real"][center_len-49:center_len+50,:,:,ch] 
			+ 1im *fid["kspace_data_z_imag"][center_len-49:center_len+50,:,:,ch]) |>  sum 
		end
		sortperm(lowk_energy)[end-k+1:end]
	end
	return top_channels
end

function get_cse_from_h5(file_path::String)
	h5open(file_path, "r") do fid
		change_array_layout(
				fid["cse_real"][:,:,:,:] + 1im *fid["cse_imag"][:,:,:,:])
	end
end

function test()
	PythonCall.GC.disable()
	Threads.@threads for i in 1:23
		a = rand(1000, 1000, 1000) .^ 2
	end
	PythonCall.GC.enable()
end

module Data
struct SubjectItem
	file_path::String
	contrast::Int
	phase_pair::Pair{Int, Int}
end
mutable struct SubjectData
	file_path::String
	id::String
	items::Union{AbstractMatrix{SubjectItem}, Nothing}
end
end