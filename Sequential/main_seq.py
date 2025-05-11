import numpy as np
import cv2
import pyswarms as ps
import random
import string
import math
import bitstring
import uuid
import time

# Implement Hiking Optimization Algorithm
class HOA:
    def __init__(self, obj_func, lb, ub, dimensions, num_hikers, max_iterations):
        """
        Initialize the Hiking Optimization Algorithm
        
        Parameters:
        -----------
        obj_func : function
            The objective function to minimize
        lb : numpy.ndarray
            Lower bounds of search space
        ub : numpy.ndarray
            Upper bounds of search space
        dimensions : int
            Number of dimensions for the problem
        num_hikers : int
            Number of hikers (population size)
        max_iterations : int
            Maximum number of iterations
        """
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dimensions = dimensions
        self.num_hikers = num_hikers
        self.max_iterations = max_iterations
        
        # Parameters for HOA
        self.angle_range = [0, 50]  # Range for angle of inclination in degrees
        self.sf_range = [1, 3]  # Range for sweep factor
        
        # Initialize best solution tracker
        self.global_best_pos = None
        self.global_best_cost = float('inf')
    
    def tobler_hiking_function(self, slope):
        """
        Calculates the walking velocity using Tobler's Hiking Function
        
        Parameters:
        -----------
        slope : float or numpy.ndarray
            Slope of the terrain (tan(θ))
            
        Returns:
        --------
        float or numpy.ndarray : velocity in km/h
        """
        return 6 * np.exp(-3.5 * np.abs(slope + 0.05))
    
    def optimize(self, init_pos=None):
        """
        Main optimization loop
        
        Parameters:
        -----------
        init_pos : numpy.ndarray, optional
            Initial positions for hikers (e.g., from PSO)
        
        Returns:
        --------
        tuple : (best position, best fitness)
        """
        # Initialize hikers' positions
        if init_pos is not None and init_pos.shape == (self.num_hikers, self.dimensions):
            hikers_pos = init_pos.copy()
        else:
            hikers_pos = np.zeros((self.num_hikers, self.dimensions))
            for i in range(self.num_hikers):
                hikers_pos[i] = self.lb + np.random.random(self.dimensions) * (self.ub - self.lb)
        
        # Evaluate initial positions
        hikers_fitness = np.array([self.obj_func(pos) for pos in hikers_pos])
        
        # Find initial best
        best_idx = np.argmin(hikers_fitness)
        self.global_best_pos = hikers_pos[best_idx].copy()
        self.global_best_cost = hikers_fitness[best_idx]
        
        # Iteration history
        cost_history = [self.global_best_cost]
        
        # Main loop
        for t in range(self.max_iterations):
            # Find the best hiker (lead hiker)
            best_idx = np.argmin(hikers_fitness)
            beta_best = hikers_pos[best_idx].copy()
            
            # Update each hiker's position
            for i in range(self.num_hikers):
                # Current position of hiker i
                beta_i_t = hikers_pos[i].copy()
                
                # Generate random angle of inclination
                theta_i_t = np.random.uniform(self.angle_range[0], self.angle_range[1]) * (np.pi / 180)  # Convert to radians
                
                # Compute slope
                s_i_t = np.tan(theta_i_t)
                
                # Compute initial velocity using Tobler's hiking function
                w_i_t_1 = self.tobler_hiking_function(s_i_t)
                
                # Random component for velocity update
                gamma_i_t = np.random.random()
                
                # Sweep factor
                alpha_i_t = np.random.uniform(self.sf_range[0], self.sf_range[1])
                
                # Determine actual velocity
                w_i_t = w_i_t_1 + gamma_i_t * (beta_best - alpha_i_t * beta_i_t)
                
                # Update position
                new_pos = beta_i_t + w_i_t
                
                # Bound the new position within search space
                new_pos = np.clip(new_pos, self.lb, self.ub)
                
                # Evaluate new position
                new_fitness = self.obj_func(new_pos)
                
                # Update hiker's position
                hikers_pos[i] = new_pos
                hikers_fitness[i] = new_fitness
                
                # Update global best if needed
                if new_fitness < self.global_best_cost:
                    self.global_best_pos = new_pos.copy()
                    self.global_best_cost = new_fitness
            
            # Store best cost of current iteration
            cost_history.append(self.global_best_cost)
        
        return self.global_best_pos, self.global_best_cost, cost_history

class ImageSteganography:
    def __init__(self, host_image_path, secret_message, output_path="stego_image.jpg", verbose=True):
        """
        Initialize the steganography class
        
        Parameters:
        -----------
        host_image_path : str
            Path to the host image
        secret_message : str
            Secret message to hide
        output_path : str
            Path to save the stego image
        verbose : bool
            If True, print detailed optimization progress
        """
        self.host_image_path = host_image_path
        self.secret_message = secret_message
        self.output_path = output_path
        self.verbose = verbose
        
        # Convert image to grayscale and get dimensions
        self.host_image = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
        if self.host_image is None:
            raise ValueError(f"Could not read image at {host_image_path}")
        
        self.height, self.width = self.host_image.shape
        
        # Convert secret message to binary
        self.secret_bits = self._message_to_bits(secret_message)
        self.secret_length = len(self.secret_bits)
        
        print(f"Host image dimensions: {self.width}x{self.height}")
        print(f"Secret message length: {len(secret_message)} characters")
        print(f"Secret bits length: {self.secret_length} bits")
        
        # Check if image has enough capacity for the message
        max_capacity = self.width * self.height  # Assuming 1 bit per pixel
        if self.secret_length > max_capacity:
            raise ValueError(f"Secret message too large for the image. Max capacity: {max_capacity} bits, Message: {self.secret_length} bits")
        
        # Optimization parameters
        self.n_particles = 10  # For PSO
        self.num_hikers = 10   # For HOA
        self.max_iterations_pso = 8 # PSO iterations
        self.max_iterations_hoa = 7 # HOA iterations
        self.c1 = 1.5  # PSO cognitive parameter
        self.c2 = 1.5  # PSO social parameter
        self.w = 0.7   # PSO inertia weight
        
        # Particle/hiker parameters
        self.directions = list(range(16))  # Direction of scanning (0-15)
        self.x_offsets = list(range(self.width))  # X-offset of starting point
        self.y_offsets = list(range(self.height))  # Y-offset of starting point
        self.bit_planes = list(range(16))  # Used LSBs for insertion (0-15)
        self.sb_pole = [0, 1]  # 0: no change, 1: complement
        self.sb_dire = [0, 1]  # 0: no change, 1: reverse
        
        # Split host image into 4 blocks
        self.block_height = self.height // 2
        self.block_width = self.width // 2
        self.blocks = self._split_image()
        
        # Split secret bits into 4 parts
        self.secret_parts = self._split_secret()
    
    def _message_to_bits(self, message):
        """Convert a string message to a bit string"""
        bits = bitstring.BitArray()
        for char in message:
            bits += bitstring.pack('uint:8', ord(char))
        return bits.bin
    
    def _bits_to_message(self, bits):
        """Convert a bit string back to a string message"""
        if len(bits) % 8 != 0:
            bits = bits.ljust((len(bits) // 8 + 1) * 8, '0')
        
        message = ""
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            message += chr(int(byte, 2))
        return message
    
    def _split_image(self):
        """Split the host image into 4 blocks (2x2)"""
        blocks = []
        for i in range(2):
            for j in range(2):
                y_start = i * self.block_height
                y_end = (i + 1) * self.block_height
                x_start = j * self.block_width
                x_end = (j + 1) * self.block_width
                block = self.host_image[y_start:y_end, x_start:x_end].copy()
                blocks.append({
                    'data': block,
                    'position': (y_start, x_start)
                })
        return blocks
    
    def _split_secret(self):
        """Split the secret bits into 4 parts"""
        bits_per_part = self.secret_length // 4
        parts = []
        for i in range(4):
            start = i * bits_per_part
            end = start + bits_per_part if i < 3 else self.secret_length
            parts.append(self.secret_bits[start:end])
        return parts
    
    def _get_scanning_sequence(self, direction, x_offset, y_offset, block_data):
        """
        Generate a sequence of pixel coordinates based on scanning direction
        """
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
        """
        Modify secret bits based on SB-Pole and SB-Dire parameters
        """
        modified_bits = secret_bits
        if sb_pole == 1:
            modified_bits = ''.join('1' if bit == '0' else '0' for bit in modified_bits)
        if sb_dire == 1:
            modified_bits = modified_bits[::-1]
        return modified_bits
    
    def _get_bit_planes_mask(self, bit_planes_value):
        """
        Get a binary mask for the specified bit planes
        """
        binary = format(bit_planes_value, '04b')
        bit_planes = []
        for i, bit in enumerate(binary):
            if bit == '1':
                bit_planes.append(i)
        if not bit_planes:
            bit_planes = [0, 1]
        return bit_planes
    
    def _embed_in_pixel(self, pixel_value, secret_bit, bit_plane):
        """
        Embed a secret bit in the specified bit plane of a pixel
        """
        binary = format(pixel_value, '08b')
        bits = list(binary)
        bits[-(bit_plane+1)] = secret_bit
        new_pixel = int(''.join(bits), 2)
        return new_pixel
    
    def _calculate_psnr(self, original, modified):
        """
        Calculate Peak Signal-to-Noise Ratio between original and modified images
        """
        mse = np.mean((original - modified) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return min(psnr, 60.0)
    
    def _calculate_ssim(self, original, modified):
        """
        Calculate Structural Similarity Index Measure between original and modified images
        """
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
        """
        Embed secret bits in a block of the host image
        """
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
    
    def _fitness_function_pso(self, particles, block_idx):
        """
        Fitness function for PSO
        """
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
                fitness[i] = 1000 * (1 - capacity_ratio)
                continue
                
            _, psnr, embedding_ratio = self._embed_in_block(block, secret_bits, params)
            capacity_penalty = (1.0 - embedding_ratio) * 100
            high_psnr_penalty = max(0, psnr - 55) * 5
            fitness[i] = -psnr + capacity_penalty + high_psnr_penalty
        
        return fitness
    
    def _fitness_function_hoa(self, params, block_idx):
        """
        Fitness function for HOA
        """
        block = self.blocks[block_idx]
        secret_bits = self.secret_parts[block_idx]
        required_capacity = len(secret_bits)
        
        direction = int(params[0]) % 16
        x_offset = int(params[1]) % self.block_width
        y_offset = int(params[2]) % self.block_height
        capacity = self._calculate_capacity(direction, x_offset, y_offset, block['data'])
        
        if capacity < required_capacity:
            capacity_ratio = capacity / required_capacity if required_capacity > 0 else 0
            return 1000 * (1 - capacity_ratio)
            
        _, psnr, embedding_ratio = self._embed_in_block(block, secret_bits, params)
        capacity_penalty = (1.0 - embedding_ratio) * 100
        high_psnr_penalty = max(0, psnr - 55) * 5
        return -psnr + capacity_penalty + high_psnr_penalty
    
    def _calculate_capacity(self, direction, x_offset, y_offset, block_data):
        """
        Calculate the embedding capacity for a given scanning configuration
        """
        coordinates = self._get_scanning_sequence(direction, x_offset, y_offset, block_data)
        return len(coordinates)
    
    def _optimize_embedding(self):
        """
        Optimize embedding using sequential PSO and HOA for each block
        """
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
            
            if self.verbose:
                print(f"\nStarting PSO optimization for Block {block_idx + 1}")
            
            # Step 1: Run PSO
            initial_positions = []
            for _ in range(self.n_particles):
                for attempt in range(10):
                    direction = random.randint(0, 15)
                    x_offset = random.randint(0, min(5, max_x))
                    y_offset = random.randint(0, min(5, max_y))
                    bit_planes = random.randint(0, 15)
                    sb_pole = random.randint(0, 1)
                    sb_dire = random.randint(0, 1)
                    capacity = self._calculate_capacity(direction, x_offset, y_offset, self.blocks[block_idx]['data'])
                    if capacity >= required_capacity:
                        initial_positions.append([direction, x_offset, y_offset, bit_planes, sb_pole, sb_dire])
                        break
                if len(initial_positions) <= _ and attempt == 9:
                    initial_positions.append([0, 0, 0, 1, 0, 0])
            
            options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}
            optimizer = ps.single.GlobalBestPSO(
                n_particles=self.n_particles,
                dimensions=6,
                options=options,
                bounds=bounds,
                init_pos=np.array(initial_positions, dtype=np.float64) if len(initial_positions) == self.n_particles else None
            )
            
            cost_function = lambda particles: self._fitness_function_pso(particles, block_idx)
            best_cost_pso, best_pos_pso = optimizer.optimize(cost_function, iters=self.max_iterations_pso)
            
            if self.verbose:
                print(f"PSO optimization completed for Block {block_idx + 1}")
                print(f"Switching to HOA optimization for Block {block_idx + 1}")
            
            # Get final PSO particle positions
            pso_final_positions = optimizer.swarm.position
            
            # Step 2: Run HOA with PSO's final positions
            hoa = HOA(
                obj_func=lambda params: self._fitness_function_hoa(params, block_idx),
                lb=lb,
                ub=ub,
                dimensions=6,
                num_hikers=self.num_hikers,
                max_iterations=self.max_iterations_hoa
            )
            
            best_pos, best_cost, _ = hoa.optimize(init_pos=pso_final_positions)
            
            if self.verbose:
                print(f"HOA optimization completed for Block {block_idx + 1}")
            
            # Verify final capacity
            direction = int(best_pos[0]) % 16
            x_offset = int(best_pos[1]) % self.block_width
            y_offset = int(best_pos[2]) % self.block_height
            final_capacity = self._calculate_capacity(direction, x_offset, y_offset, self.blocks[block_idx]['data'])
            
            print(f"Block {block_idx+1} optimization complete:")
            print(f"Best parameters: {best_pos}")
            print(f"Best fitness: {-best_cost}")
            print(f"Required capacity: {required_capacity}, Final capacity: {final_capacity}")
            
            if final_capacity < required_capacity:
                print(f"WARNING: Best solution for Block {block_idx+1} still has insufficient capacity!")
                for test_direction in range(16):
                    test_capacity = self._calculate_capacity(test_direction, 0, 0, self.blocks[block_idx]['data'])
                    if test_capacity >= required_capacity:
                        best_pos[0] = test_direction
                        best_pos[1] = 0
                        best_pos[2] = 0
                        print(f"Forced direction {test_direction} with capacity {test_capacity}")
                        break
            
            optimal_params.append(best_pos)
        
        return optimal_params
    
    def embed(self):
        """
        Embed secret message in host image
        """
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
        print(f"Bits per change ratio: {total_bits/total_changes if total_changes > 0 else 'Infinity'}")
        
        cv2.imwrite(self.output_path, stego_image)
        print(f"Stego image saved to {self.output_path}")
        
        return stego_image, overall_psnr, optimal_params

    def extract(self, stego_image, params, message_length):
        """
        Extract hidden message from stego image
        """
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
                stego_blocks.append({
                    'data': block,
                    'position': (y_start, x_start)
                })
        
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
    output_path = "stego_sail_seq.jpg"
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
    
    stego = ImageSteganography(host_image_path, secret_message, output_path, verbose=True)
    stego_image, psnr, params = stego.embed()
    
    print("Steganography complete!")
    print(f"PSNR: {psnr:.2f}")
    print(f"Optimal parameters: {params}")
    
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