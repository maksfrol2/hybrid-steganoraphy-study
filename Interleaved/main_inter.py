import numpy as np
import cv2
from PIL import Image
import pyswarms as ps
import random
import string
import math
import bitstring
import time

# Hiking Optimization Algorithm
class HOA:
    def __init__(self, obj_func, lb, ub, dimensions, num_hikers, max_iterations):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dimensions = dimensions
        self.num_hikers = num_hikers
        self.max_iterations = max_iterations
        self.angle_range = [0, 50]
        self.sf_range = [1, 3]
        self.global_best_pos = None
        self.global_best_cost = float('inf')
    
    def tobler_hiking_function(self, slope):
        return 6 * np.exp(-3.5 * np.abs(slope + 0.05))
    
    def optimize(self):
        hikers_pos = np.zeros((self.num_hikers, self.dimensions))
        for i in range(self.num_hikers):
            hikers_pos[i] = self.lb + np.random.random(self.dimensions) * (self.ub - self.lb)
        
        hikers_fitness = np.array([self.obj_func(pos) for pos in hikers_pos])
        best_idx = np.argmin(hikers_fitness)
        self.global_best_pos = hikers_pos[best_idx].copy()
        self.global_best_cost = hikers_fitness[best_idx]
        
        for t in range(self.max_iterations):
            best_idx = np.argmin(hikers_fitness)
            beta_best = hikers_pos[best_idx].copy()
            
            for i in range(self.num_hikers):
                beta_i_t = hikers_pos[i].copy()
                theta_i_t = np.random.uniform(self.angle_range[0], self.angle_range[1]) * (np.pi / 180)
                s_i_t = np.tan(theta_i_t)
                w_i_t_1 = self.tobler_hiking_function(s_i_t)
                gamma_i_t = np.random.random()
                alpha_i_t = np.random.uniform(self.sf_range[0], self.sf_range[1])
                w_i_t = w_i_t_1 + gamma_i_t * (beta_best - alpha_i_t * beta_i_t)
                new_pos = beta_i_t + w_i_t
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self.obj_func(new_pos)
                hikers_pos[i] = new_pos
                hikers_fitness[i] = new_fitness
                if new_fitness < self.global_best_cost:
                    self.global_best_pos = new_pos.copy()
                    self.global_best_cost = new_fitness
        
        return self.global_best_pos, self.global_best_cost

class ImageSteganography:
    def __init__(self, host_image_path, secret_message, output_path="stego_image.jpg"):
        self.host_image_path = host_image_path
        self.secret_message = secret_message
        self.output_path = output_path
        self.host_image = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
        if self.host_image is None:
            raise ValueError(f"Could not read image at {host_image_path}")
        
        self.height, self.width = self.host_image.shape
        self.secret_bits = self._message_to_bits(secret_message)
        self.secret_length = len(self.secret_bits)
        
        print(f"Host image dimensions: {self.width}x{self.height}")
        print(f"Secret message length: {len(secret_message)} characters")
        print(f"Secret bits length: {self.secret_length} bits")
        
        max_capacity = self.width * self.height
        if self.secret_length > max_capacity:
            raise ValueError(f"Secret message too large for the image. Max capacity: {max_capacity} bits, Message: {self.secret_length} bits")
        
        self.n_particles = 10
        self.max_iterations = 15
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.directions = list(range(16))
        self.x_offsets = list(range(self.width))
        self.y_offsets = list(range(self.height))
        self.bit_planes = list(range(16))
        self.sb_pole = [0, 1]
        self.sb_dire = [0, 1]
        self.block_height = self.height // 2
        self.block_width = self.width // 2
        self.blocks = self._split_image()
        self.secret_parts = self._split_secret()
    
    def _message_to_bits(self, message):
        bits = bitstring.BitArray()
        for char in message:
            bits += bitstring.pack('uint:8', ord(char))
        return bits.bin
    
    def _bits_to_message(self, bits):
        if len(bits) % 8 != 0:
            bits = bits.ljust((len(bits) // 8 + 1) * 8, '0')
        message = ""
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            message += chr(int(byte, 2))
        return message
    
    def _split_image(self):
        blocks = []
        for i in range(2):
            for j in range(2):
                y_start = i * self.block_height
                y_end = (i + 1) * self.block_height
                x_start = j * self.block_width
                x_end = (j + 1) * self.block_width
                block = self.host_image[y_start:y_end, x_start:x_end].copy()
                blocks.append({'data': block, 'position': (y_start, x_start)})
        return blocks
    
    def _split_secret(self):
        bits_per_part = self.secret_length // 4
        parts = []
        for i in range(4):
            start = i * bits_per_part
            end = start + bits_per_part if i < 3 else self.secret_length
            parts.append(self.secret_bits[start:end])
        return parts
    
    def _get_scanning_sequence(self, direction, x_offset, y_offset, block_data):
        bh, bw = block_data.shape
        x_offset = min(x_offset, bw - 1)
        y_offset = min(y_offset, bh - 1)
        scanning_properties = {
            0: {'rows': 'top_to_bottom', 'cols': 'left_to_right', 'type': 'triangle', 'arrangement': 'cols_then_rows'},
            1: {'rows': 'top_to_bottom', 'cols': 'right_to_left', 'type': 'triangle', 'arrangement': 'cols_then_rows'},
            2: {'rows': 'bottom_to_top', 'cols': 'left_to_right', 'type': 'triangle', 'arrangement': 'cols_then_rows'},
            3: {'rows': 'bottom_to_top', 'cols': 'right_to_left', 'type': 'triangle', 'arrangement': 'cols_then_rows'},
            4: {'rows': 'top_to_bottom', 'cols': 'left_to_right', 'type': 'square', 'arrangement': 'cols_then_rows'},
            5: {'rows': 'top_to_bottom', 'cols': 'right_to_left', 'type': 'square', 'arrangement': 'cols_then_rows'},
            6: {'rows': 'bottom_to_top', 'cols': 'left_to_right', 'type': 'square', 'arrangement': 'cols_then_rows'},
            7: {'rows': 'bottom_to_top', 'cols': 'right_to_left', 'type': 'square', 'arrangement': 'cols_then_rows'},
            8: {'rows': 'top_to_bottom', 'cols': 'left_to_right', 'type': 'triangle', 'arrangement': 'rows_then_cols'},
            9: {'rows': 'top_to_bottom', 'cols': 'right_to_left', 'type': 'triangle', 'arrangement': 'rows_then_cols'},
            10: {'rows': 'bottom_to_top', 'cols': 'left_to_right', 'type': 'triangle', 'arrangement': 'rows_then_cols'},
            11: {'rows': 'bottom_to_top', 'cols': 'right_to_left', 'type': 'triangle', 'arrangement': 'rows_then_cols'},
            12: {'rows': 'top_to_bottom', 'cols': 'left_to_right', 'type': 'square', 'arrangement': 'rows_then_cols'},
            13: {'rows': 'top_to_bottom', 'cols': 'right_to_left', 'type': 'square', 'arrangement': 'rows_then_cols'},
            14: {'rows': 'bottom_to_top', 'cols': 'left_to_right', 'type': 'square', 'arrangement': 'rows_then_cols'},
            15: {'rows': 'bottom_to_top', 'cols': 'right_to_left', 'type': 'square', 'arrangement': 'rows_then_cols'}
        }
        props = scanning_properties[direction]
        if props['rows'] == 'top_to_bottom':
            rows = list(range(y_offset, bh))
        else:
            rows = list(range(y_offset, -1, -1))
        if props['cols'] == 'left_to_right':
            cols = list(range(x_offset, bw))
        else:
            cols = list(range(x_offset, -1, -1))
        coordinates = []
        if props['type'] == 'triangle':
            all_points = []
            if props['arrangement'] == 'cols_then_rows':
                all_points = [(y, x) for y in rows for x in cols]
                if props['cols'] == 'left_to_right':
                    sorted_points = sorted(all_points, key=lambda p: p[0] + p[1])
                else:
                    sorted_points = sorted(all_points, key=lambda p: p[0] - p[1])
                coordinates = sorted_points
            else:
                all_points = [(y, x) for x in cols for y in rows]
                if props['cols'] == 'left_to_right':
                    sorted_points = sorted(all_points, key=lambda p: p[1] + p[0])
                else:
                    sorted_points = sorted(all_points, key=lambda p: p[1] - p[0])
                coordinates = sorted_points
        else:
             if props['arrangement'] == 'cols_then_rows':
                 for y in rows:
                     for x in cols:
                         coordinates.append((y, x))
             else:
                 for x in cols:
                     for y in rows:
                         coordinates.append((y, x))
        return coordinates

    def _modify_secret_bits(self, secret_bits, sb_pole, sb_dire):
        modified_bits = secret_bits
        if sb_pole == 1:
            modified_bits = ''.join('1' if bit == '0' else '0' for bit in modified_bits)
        if sb_dire == 1:
            modified_bits = modified_bits[::-1]
        return modified_bits
    
    def _get_bit_planes_mask(self, bit_planes_value):
        binary = format(bit_planes_value, '04b')
        bit_planes = []
        for i, bit in enumerate(binary):
            if bit == '1':
                bit_planes.append(i)
        if not bit_planes:
            bit_planes = [0, 1]
        return bit_planes
    
    def _embed_in_pixel(self, pixel_value, secret_bit, bit_plane):
        binary = format(pixel_value, '08b')
        bits = list(binary)
        bits[-(bit_plane+1)] = secret_bit
        new_pixel = int(''.join(bits), 2)
        return new_pixel
    
    def _calculate_psnr(self, original, modified):
        mse = np.mean((original - modified) ** 2)
        if mse == 0:
            return 0  # Return 0 for infinite PSNR
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return psnr
    
    def _calculate_ssim(self, original, modified):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        original = original.astype(np.float64)
        modified = modified.astype(np.float64)
        mu_original = np.mean(original)
        mu_modified = np.mean(modified)
        sigma_original_sq = np.var(original)
        sigma_modified_sq = np.var(modified)
        sigma_cross = np.mean((original - mu_original) * (modified - mu_modified))
        numerator = (2 * mu_original * mu_modified + C1) * (2 * sigma_cross + C2)
        denominator = (mu_original**2 + mu_modified**2 + C1) * (sigma_original_sq + sigma_modified_sq + C2)
        ssim = numerator / denominator
        return ssim
    
    def _embed_in_block(self, block, secret_bits, params):
        direction = int(params[0]) % 16
        x_offset = int(params[1]) % self.block_width
        y_offset = int(params[2]) % self.block_height
        bit_planes_value = int(params[3]) % 16
        sb_pole = int(params[4]) % 2
        sb_dire = int(params[5]) % 2
        modified_bits = self._modify_secret_bits(secret_bits, sb_pole, sb_dire)
        bit_planes = self._get_bit_planes_mask(bit_planes_value)
        coordinates = self._get_scanning_sequence(direction, x_offset, y_offset, block['data'])
        if len(coordinates) < len(modified_bits):
            print(f"Warning: Not enough capacity to embed all secret bits. Only {len(coordinates)}/{len(modified_bits)} bits will be embedded.")
        embedding_ratio = min(1.0, len(coordinates) / len(modified_bits))
        coordinates = coordinates[:len(modified_bits)]
        modified_block = block['data'].copy()
        bits_embedded = 0
        for i, (y, x) in enumerate(coordinates):
            if i >= len(modified_bits):
                break
            secret_bit = modified_bits[i]
            bit_plane = bit_planes[i % len(bit_planes)]
            current_bit = format(modified_block[y, x], '08b')[-(bit_plane+1)]
            if current_bit != secret_bit:
                modified_block[y, x] = self._embed_in_pixel(modified_block[y, x], secret_bit, bit_plane)
                bits_embedded += 1
        if bits_embedded == 0 and len(coordinates) > 0:
            y, x = coordinates[0]
            modified_block[y, x] = self._embed_in_pixel(modified_block[y, x], '1', 0)
            bits_embedded = 1
        psnr = self._calculate_psnr(block['data'], modified_block)
        return {'data': modified_block, 'position': block['position']}, psnr, embedding_ratio
    
    def _fitness_function(self, particles, block_idx):
        # Check if particles is a 1D array (single parameter set) or 2D array (multiple particles)
        if particles.ndim == 1:
            # Single parameter set (used by HOA)
            params = particles
            direction = int(params[0]) % 16
            x_offset = int(params[1]) % self.block_width
            y_offset = int(params[2]) % self.block_height
            block = self.blocks[block_idx]
            secret_bits = self.secret_parts[block_idx]
            required_capacity = len(secret_bits)
            capacity = self._calculate_capacity(direction, x_offset, y_offset, block['data'])
            if capacity < required_capacity:
                capacity_ratio = capacity / required_capacity if required_capacity > 0 else 0
                fitness_value = 1000 * (1 - capacity_ratio)
            else:
                _, psnr, embedding_ratio = self._embed_in_block(block, secret_bits, params)
                capacity_penalty = (1.0 - embedding_ratio) * 100
                high_psnr_penalty = max(0, psnr - 55) * 5
                fitness_value = -psnr + capacity_penalty + high_psnr_penalty
            return fitness_value
        else:
            # Multiple particles (used by PSO)
            n_particles = particles.shape[0]
            fitness = np.zeros(n_particles)
            block = self.blocks[block_idx]
            secret_bits = self.secret_parts[block_idx]
            required_capacity = len(secret_bits)
            for i in range(n_particles):
                params = particles[i]
                direction = int(params[0]) % 16
                x_offset = int(params[1]) % self.block_width
                y_offset = int(params[2]) % self.block_height
                capacity = self._calculate_capacity(direction, x_offset, y_offset, block['data'])
                if capacity < required_capacity:
                    capacity_ratio = capacity / required_capacity if required_capacity > 0 else 0
                    fitness_value = 1000 * (1 - capacity_ratio)
                else:
                    _, psnr, embedding_ratio = self._embed_in_block(block, secret_bits, params)
                    capacity_penalty = (1.0 - embedding_ratio) * 100
                    high_psnr_penalty = max(0, psnr - 55) * 5
                    fitness_value = -psnr + capacity_penalty + high_psnr_penalty
                fitness[i] = fitness_value
            return fitness
    
    def _calculate_capacity(self, direction, x_offset, y_offset, block_data):
        coordinates = self._get_scanning_sequence(direction, x_offset, y_offset, block_data)
        return len(coordinates)
    
    def _optimize_embedding(self):
        optimal_params = []
        min_x = 0
        min_y = 0
        max_x = max(0, self.block_width - 1)
        max_y = max(0, self.block_height - 1)
        lb = np.array([0.0, min_x, min_y, 0.0, 0.0, 0.0], dtype=np.float64)
        ub = np.array([15.0, max_x, max_y, 15.0, 1.0, 1.0], dtype=np.float64)
        bounds = (lb, ub)
        
        for block_idx in range(4):
            secret_bits = self.secret_parts[block_idx]
            required_capacity = len(secret_bits)
            options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}
            current_algorithm = 'PSO'
            iteration = 0
            prev_best_psnr = -float('inf')
            best_params = None
            best_fitness = float('inf')
            
            while iteration < self.max_iterations:
                print(f"Block {block_idx+1}, Iteration {iteration+1}, Algorithm: {current_algorithm}")
                
                if current_algorithm == 'PSO':
                    optimizer = ps.single.GlobalBestPSO(
                        n_particles=self.n_particles,
                        dimensions=6,
                        options=options,
                        bounds=bounds
                    )
                    cost_function = lambda particles: self._fitness_function(particles, block_idx)
                    best_cost, best_pos = optimizer.optimize(cost_function, iters=1,verbose=False)
                    _, current_psnr, _ = self._embed_in_block(self.blocks[block_idx], secret_bits, best_pos)
                    print(f"PSO Best PSNR: {current_psnr:.2f}")
                    
                    if current_psnr <= prev_best_psnr:
                        current_algorithm = 'HOA'
                        print(f"Migration: PSO -> HOA (PSNR did not increase: {current_psnr:.2f} <= {prev_best_psnr:.2f})")
                    else:
                        prev_best_psnr = current_psnr
                        if best_cost < best_fitness:
                            best_params = best_pos.copy()
                            best_fitness = best_cost
                
                else:  # HOA
                    def cost_function(params):
                        return self._fitness_function(params, block_idx)
                    hoa = HOA(
                        obj_func=cost_function,
                        lb=lb,
                        ub=ub,
                        dimensions=6,
                        num_hikers=self.n_particles,
                        max_iterations=1
                    )
                    best_pos, best_cost = hoa.optimize()
                    _, current_psnr, _ = self._embed_in_block(self.blocks[block_idx], secret_bits, best_pos)
                    print(f"HOA Best PSNR: {current_psnr:.2f}")
                    
                    if current_psnr <= prev_best_psnr:
                        current_algorithm = 'PSO'
                        print(f"Migration: HOA -> PSO (PSNR did not increase: {current_psnr:.2f} <= {prev_best_psnr:.2f})")
                    else:
                        prev_best_psnr = current_psnr
                        if best_cost < best_fitness:
                            best_params = best_pos.copy()
                            best_fitness = best_cost
                
                iteration += 1
            
            direction = int(best_params[0]) % 16
            x_offset = int(best_params[1]) % self.block_width
            y_offset = int(best_params[2]) % self.block_height
            final_capacity = self._calculate_capacity(direction, x_offset, y_offset, self.blocks[block_idx]['data'])
            print(f"Block {block_idx+1} optimization complete:")
            print(f"Best parameters: {best_params}")
            print(f"Best fitness: {-best_fitness}")
            print(f"Required capacity: {required_capacity}, Final capacity: {final_capacity}")
            
            if final_capacity < required_capacity:
                for test_direction in range(16):
                    test_capacity = self._calculate_capacity(test_direction, 0, 0, self.blocks[block_idx]['data'])
                    if test_capacity >= required_capacity:
                        best_params[0] = test_direction
                        best_params[1] = 0
                        best_params[2] = 0
                        break
            
            optimal_params.append(best_params)
        
        return optimal_params
    
    def embed(self):
        optimal_params = self._optimize_embedding()
        stego_image = self.host_image.copy()
        total_bits = 0
        total_changes = 0
        block_psnrs = []
        block_ssims = []
        
        for block_idx in range(4):
            block = self.blocks[block_idx]
            secret_bits = self.secret_parts[block_idx]
            params = optimal_params[block_idx]
            original_block = block['data'].copy()
            modified_block, psnr, embedding_ratio = self._embed_in_block(block, secret_bits, params)
            block_psnrs.append(psnr)
            ssim = self._calculate_ssim(original_block, modified_block['data'])
            block_ssims.append(ssim)
            y_start, x_start = block['position']
            y_end = y_start + self.block_height
            x_end = x_start + self.block_width
            stego_image[y_start:y_end, x_start:x_end] = modified_block['data']
            diff = np.sum(original_block != modified_block['data'])
            total_changes += diff
            total_bits += len(secret_bits)
            print(f"Block {block_idx+1} embedded with PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, Embedding ratio: {embedding_ratio*100:.2f}%")
            print(f"Pixel changes in Block {block_idx+1}: {diff} out of {original_block.size} pixels")
        
        overall_psnr = self._calculate_psnr(self.host_image, stego_image)
        overall_ssim = self._calculate_ssim(self.host_image, stego_image)
        print(f"Overall PSNR: {overall_psnr:.2f}")
        print(f"Overall SSIM: {overall_ssim:.4f}")
        print(f"Average block PSNR: {sum(block_psnrs)/len(block_psnrs):.2f}")
        print(f"Average block SSIM: {sum(block_ssims)/len(block_ssims):.4f}")
        print(f"Total bit changes: {total_changes} out of {self.host_image.size} pixels ({total_changes/self.host_image.size*100:.2f}%)")
        cv2.imwrite(self.output_path, stego_image)
        print(f"Stego image saved to {self.output_path}")
        return stego_image, overall_psnr, optimal_params

    def extract(self, stego_image, params, message_length):
        bits_length = message_length * 8
        stego_height, stego_width = stego_image.shape
        block_height = stego_height // 2
        block_width = stego_width // 2
        stego_blocks = []
        for i in range(2):
            for j in range(2):
                y_start = i * block_height
                y_end = (i + 1) * block_height
                x_start = j * block_width
                x_end = (j + 1) * block_width
                block = stego_image[y_start:y_end, x_start:x_end].copy()
                stego_blocks.append({'data': block, 'position': (y_start, x_start)})
        bits_per_block = bits_length // 4
        extracted_bits = ""
        for block_idx in range(4):
            block = stego_blocks[block_idx]
            block_params = params[block_idx]
            direction = int(block_params[0]) % 16
            x_offset = int(block_params[1]) % block_width
            y_offset = int(block_params[2]) % block_height
            bit_planes_value = int(block_params[3]) % 16
            sb_pole = int(block_params[4]) % 2
            sb_dire = int(block_params[5]) % 2
            bit_planes = self._get_bit_planes_mask(bit_planes_value)
            coordinates = self._get_scanning_sequence(direction, x_offset, y_offset, block['data'])
            block_bits = ""
            bits_to_extract = bits_per_block if block_idx < 3 else (bits_length - block_idx * bits_per_block)
            for i in range(min(bits_to_extract, len(coordinates))):
                y, x = coordinates[i]
                pixel_value = block['data'][y, x]
                bit_plane = bit_planes[i % len(bit_planes)]
                binary = format(pixel_value, '08b')
                extracted_bit = binary[-(bit_plane+1)]
                block_bits += extracted_bit
            if sb_dire == 1:
                block_bits = block_bits[::-1]
            if sb_pole == 1:
                block_bits = ''.join('1' if bit == '0' else '0' for bit in block_bits)
            extracted_bits += block_bits
        extracted_message = self._bits_to_message(extracted_bits[:bits_length])
        return extracted_message

def main():
    host_image_path = "./images/sail.tiff"
    output_path = "steg_sail_inter.jpg"
    message_file = "./message_20k.txt"
    try:
        with open(message_file, 'r', encoding='utf-8') as file:
            secret_message = file.read()
            print(f"Secret message read from {message_file}")
            print(f"Message length: {len(secret_message)} characters")
    except FileNotFoundError:
        print(f"Error: File {message_file} not found.")
        secret_message = "This is a secret message for testing steganography."
        print(f"Using default message: '{secret_message}'")
    except Exception as e:
        print(f"Error reading message file: {str(e)}")
        return
    stego = ImageSteganography(host_image_path, secret_message, output_path)
    stego_image, psnr, params = stego.embed()
    print("Steganography complete!")
    print(f"PSNR: {psnr:.2f}")
    try:
        extracted_message = stego.extract(stego_image, params, len(secret_message))
        print(f"Extracted message: '{extracted_message[:100]}...'")
        if extracted_message == secret_message:
            print("Extraction successful! The messages match.")
        else:
            print("Extraction failed. The messages don't match.")
            min_len = min(len(secret_message), len(extracted_message))
            correct_chars = sum(1 for a, b in zip(secret_message[:min_len], extracted_message[:min_len]) if a == b)
            accuracy = correct_chars / len(secret_message) * 100 if len(secret_message) > 0 else 0
            print(f"Extraction accuracy: {accuracy:.2f}%")
            if min_len > 0:
                for i in range(min(100, min_len)):
                    if secret_message[i] != extracted_message[i]:
                        print(f"First difference at position {i}: '{secret_message[i]}' vs '{extracted_message[i]}'")
                        break
    except Exception as e:
        print(f"Error during extraction: {str(e)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))