import numpy as np
import cv2
from PIL import Image
import pyswarms as ps
import random
import string
import math
import bitstring
import time

class ImageSteganography:
    def __init__(self, host_image_path, secret_message, output_path="stego_image.jpg"):
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
        """
        self.host_image_path = host_image_path
        self.secret_message = secret_message
        self.output_path = output_path
        
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
        
        # PSO parameters
        self.n_particles = 20
        self.max_iterations = 20
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w = 0.7   # Inertia weight
        
        # Particle parameters
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
        # Convert each character to 8 bits
        for char in message:
            bits += bitstring.pack('uint:8', ord(char))
        return bits.bin
    
    def _bits_to_message(self, bits):
        """Convert a bit string back to a string message"""
        # Ensure bits length is multiple of 8
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
        
        Parameters:
        -----------
        direction : int
            Scanning direction (0-15)
        x_offset : int
            X-offset of starting point
        y_offset : int
            Y-offset of starting point
        block_data : numpy.ndarray
            Block image data
        
        Returns:
        --------
        list
            List of (y, x) coordinates for pixels
        """
        bh, bw = block_data.shape
        
        # Ensure offsets are within block boundaries
        x_offset = min(x_offset, bw - 1)
        y_offset = min(y_offset, bh - 1)
        
        # Define scanning properties based on direction value
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
        
        # Generate row indices
        if props['rows'] == 'top_to_bottom':
            rows = list(range(y_offset, bh))
        else:  # bottom_to_top
            rows = list(range(y_offset, -1, -1))
        
        # Generate column indices
        if props['cols'] == 'left_to_right':
            cols = list(range(x_offset, bw))
        else:  # right_to_left
            cols = list(range(x_offset, -1, -1))
        
        # Generate coordinate sequence based on arrangement and type
        coordinates = []
        
        if props['type'] == 'triangle':
            # Triangle pattern implementation - creates a diagonal-like traversal
            all_points = []
            
            if props['arrangement'] == 'cols_then_rows':
                # Generate all points
                all_points = [(y, x) for y in rows for x in cols]
                
                # Sort points based on diagonal distance
                if props['cols'] == 'left_to_right':
                    # For left-to-right, sort by y+x to get top-left to bottom-right diagonal
                    sorted_points = sorted(all_points, key=lambda p: p[0] + p[1])
                else:
                    # For right-to-left, sort by y-x to get top-right to bottom-left diagonal
                    sorted_points = sorted(all_points, key=lambda p: p[0] - p[1])
                
                coordinates = sorted_points
            else:  # rows_then_cols
                # Similar but prioritizing row-wise traversal first
                all_points = [(y, x) for x in cols for y in rows]
                
                if props['cols'] == 'left_to_right':
                    sorted_points = sorted(all_points, key=lambda p: p[1] + p[0])
                else:
                    sorted_points = sorted(all_points, key=lambda p: p[1] - p[0])
                
                coordinates = sorted_points
        else:  # Square pattern - standard row/column traversal
            if props['arrangement'] == 'cols_then_rows':
                for y in rows:
                    for x in cols:
                        coordinates.append((y, x))
            else:  # rows_then_cols
                for x in cols:
                    for y in rows:
                        coordinates.append((y, x))
        
        return coordinates
    
    def _modify_secret_bits(self, secret_bits, sb_pole, sb_dire):
        """
        Modify secret bits based on SB-Pole and SB-Dire parameters
        
        Parameters:
        -----------
        secret_bits : str
            Secret bits to modify
        sb_pole : int
            0: no change, 1: complement
        sb_dire : int
            0: no change, 1: reverse
        
        Returns:
        --------
        str
            Modified secret bits
        """
        modified_bits = secret_bits
        
        # Apply SB-Pole: complement bits if sb_pole is 1
        if sb_pole == 1:
            modified_bits = ''.join('1' if bit == '0' else '0' for bit in modified_bits)
        
        # Apply SB-Dire: reverse bit order if sb_dire is 1
        if sb_dire == 1:
            modified_bits = modified_bits[::-1]
        
        return modified_bits
    
    def _get_bit_planes_mask(self, bit_planes_value):
        """
        Get a binary mask for the specified bit planes
        
        Parameters:
        -----------
        bit_planes_value : int
            Value specifying which bit planes to use (0-15)
        
        Returns:
        --------
        list
            List of bit positions (0-7) to use
        """
        # Convert bit_planes_value to 4-bit binary
        binary = format(bit_planes_value, '04b')
        
        # Map to bit planes (0-3 are LSBs)
        bit_planes = []
        for i, bit in enumerate(binary):
            if bit == '1':
                # Use multiple bit planes to ensure data is actually embedded
                # This will reduce PSNR but improve embedding
                bit_planes.append(i)
        
        # Ensure at least one bit plane is selected
        if not bit_planes:
            # Default to using both bit plane 0 and 1 to ensure visible change
            bit_planes = [0, 1]
            
        return bit_planes
    
    def _embed_in_pixel(self, pixel_value, secret_bit, bit_plane):
        """
        Embed a secret bit in the specified bit plane of a pixel
        
        Parameters:
        -----------
        pixel_value : int
            Pixel value (0-255)
        secret_bit : str
            Secret bit ('0' or '1')
        bit_plane : int
            Bit plane to use (0-7, where 0 is LSB)
        
        Returns:
        --------
        int
            Modified pixel value
        """
        # Convert pixel to binary (8 bits)
        binary = format(pixel_value, '08b')
        
        # Create a list for easier manipulation
        bits = list(binary)
        
        # Replace the bit at specified position
        bits[-(bit_plane+1)] = secret_bit
        
        # Convert back to integer
        new_pixel = int(''.join(bits), 2)
        
        return new_pixel
    
    def _calculate_psnr(self, original, modified):
        """
        Calculate Peak Signal-to-Noise Ratio between original and modified images
        
        Parameters:
        -----------
        original : numpy.ndarray
            Original image
        modified : numpy.ndarray
            Modified image
        
        Returns:
        --------
        float
            PSNR value
        """
        mse = np.mean((original - modified) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        
        # Cap PSNR at a reasonable value to prevent optimization issues
        return min(psnr, 60.0)
        
    def _calculate_ssim(self, original, modified):
        """
        Calculate Structural Similarity Index Measure between original and modified images
        
        Parameters:
        -----------
        original : numpy.ndarray
            Original image
        modified : numpy.ndarray
            Modified image
        
        Returns:
        --------
        float
            SSIM value between 0 and 1 (1 means identical images)
        """
        # Constants for SSIM calculation
        C1 = (0.01 * 255)**2  # Constant to stabilize division when denominator is weak
        C2 = (0.03 * 255)**2  # Constant to stabilize division when denominator is weak
        
        # Convert images to float
        original = original.astype(np.float64)
        modified = modified.astype(np.float64)
        
        # Calculate means
        mu_original = np.mean(original)
        mu_modified = np.mean(modified)
        
        # Calculate variances and covariance
        sigma_original_sq = np.var(original)
        sigma_modified_sq = np.var(modified)
        sigma_cross = np.mean((original - mu_original) * (modified - mu_modified))
        
        # Calculate SSIM
        numerator = (2 * mu_original * mu_modified + C1) * (2 * sigma_cross + C2)
        denominator = (mu_original**2 + mu_modified**2 + C1) * (sigma_original_sq + sigma_modified_sq + C2)
        ssim = numerator / denominator
        
        return ssim
    
    def _embed_in_block(self, block, secret_bits, params):
        """
        Embed secret bits in a block of the host image
        
        Parameters:
        -----------
        block : dict
            Block information
        secret_bits : str
            Secret bits to embed
        params : list
            PSO parameters [direction, x_offset, y_offset, bit_planes, sb_pole, sb_dire]
        
        Returns:
        --------
        tuple
            (Modified block, PSNR value, embedding_ratio)
        """
        direction = int(params[0]) % 16
        x_offset = int(params[1]) % self.block_width
        y_offset = int(params[2]) % self.block_height
        bit_planes_value = int(params[3]) % 16
        sb_pole = int(params[4]) % 2
        sb_dire = int(params[5]) % 2
        
        # Modify secret bits based on sb_pole and sb_dire
        modified_bits = self._modify_secret_bits(secret_bits, sb_pole, sb_dire)
        
        # Get bit planes to use
        bit_planes = self._get_bit_planes_mask(bit_planes_value)
        
        # Get scanning sequence
        coordinates = self._get_scanning_sequence(direction, x_offset, y_offset, block['data'])
        
        # Check if we have enough capacity
        if len(coordinates) < len(modified_bits):
            print(f"Warning: Not enough capacity to embed all secret bits. Only {len(coordinates)}/{len(modified_bits)} bits will be embedded.")
            # This block will have a poor fitness score
        
        # Calculate embedding ratio
        embedding_ratio = min(1.0, len(coordinates) / len(modified_bits))
        
        # Limit coordinates to number of secret bits
        coordinates = coordinates[:len(modified_bits)]
        
        # Create a copy of the block for modification
        modified_block = block['data'].copy()
        
        # Embed secret bits in block
        bits_embedded = 0
        for i, (y, x) in enumerate(coordinates):
            if i >= len(modified_bits):
                break
                
            # Get the bit to embed
            secret_bit = modified_bits[i]
            
            # Get the bit plane to use for this bit
            bit_plane = bit_planes[i % len(bit_planes)]
            
            # Only apply change if the bit is different
            current_bit = format(modified_block[y, x], '08b')[-(bit_plane+1)]
            if current_bit != secret_bit:
                modified_block[y, x] = self._embed_in_pixel(modified_block[y, x], secret_bit, bit_plane)
                bits_embedded += 1
        
        # If no bits were embedded, force a change to ensure message is actually hidden
        if bits_embedded ==  0:
            if len(coordinates) > 0:
                y, x = coordinates[0]
                # Modify the LSB
                modified_block[y, x] = self._embed_in_pixel(modified_block[y, x], '1', 0)
                bits_embedded = 1
        
        # Calculate PSNR
        psnr = self._calculate_psnr(block['data'], modified_block)
        
        return {'data': modified_block, 'position': block['position']}, psnr, embedding_ratio
    
    def _fitness_function(self, particles, block_idx):
        """
        Fitness function for PSO
        
        Parameters:
        -----------
        particles : numpy.ndarray
            Particle positions
        block_idx : int
            Block index (0-3)
        
        Returns:
        --------
        numpy.ndarray
            Fitness values (negative PSNR with penalty for low embedding ratio)
        """
        n_particles = particles.shape[0]
        fitness = np.zeros(n_particles)
        
        block = self.blocks[block_idx]
        secret_bits = self.secret_parts[block_idx]
        required_capacity = len(secret_bits)
        
        for i in range(n_particles):
            params = particles[i]
            
            # Calculate capacity before embedding to check if it's sufficient
            direction = int(params[0]) % 16
            x_offset = int(params[1]) % self.block_width
            y_offset = int(params[2]) % self.block_height
            
            capacity = self._calculate_capacity(direction, x_offset, y_offset, block['data'])
            
            # If capacity is insufficient, apply a severe penalty
            if capacity < required_capacity:
                # Penalty proportional to how much capacity is missing
                capacity_ratio = capacity / required_capacity if required_capacity > 0 else 0
                # Large penalty to discourage solutions with insufficient capacity
                fitness[i] = 1000 * (1 - capacity_ratio)
                continue
                
            # Only perform embedding if capacity is sufficient
            _, psnr, embedding_ratio = self._embed_in_block(block, secret_bits, params)
            
            # Penalty for low embedding ratio
            capacity_penalty = (1.0 - embedding_ratio) * 100
            
            # If PSNR is too high, apply a penalty to encourage meaningful embedding
            high_psnr_penalty = max(0, psnr - 55) * 5
            
            # Negative PSNR as we want to maximize PSNR but PSO minimizes
            # Add penalties to balance PSNR and embedding capacity
            fitness[i] = -psnr + capacity_penalty + high_psnr_penalty
        
        return fitness
    
    def _calculate_capacity(self, direction, x_offset, y_offset, block_data):
        """
        Calculate the embedding capacity for a given scanning configuration
        
        Parameters:
        -----------
        direction : int
            Scanning direction (0-15)
        x_offset : int
            X-offset of starting point
        y_offset : int
            Y-offset of starting point
        block_data : numpy.ndarray
            Block image data
            
        Returns:
        --------
        int
            Number of pixels available for embedding
        """
        coordinates = self._get_scanning_sequence(direction, x_offset, y_offset, block_data)
        return len(coordinates)
        
    def _optimize_embedding(self):
        """
        Optimize embedding using PSO for each block
        
        Returns:
        --------
        list
            Optimal parameters for each block
        """
        optimal_params = []
        
        # Define bounds for particles with safety checks
        min_x = 0
        min_y = 0
        max_x = max(0, self.block_width - 1)
        max_y = max(0, self.block_height - 1)
        
        # Using float data types for bounds to avoid casting issues
        lb = np.array([0.0, min_x, min_y, 0.0, 0.0, 0.0], dtype=np.float64)
        ub = np.array([15.0, max_x, max_y, 15.0, 1.0, 1.0], dtype=np.float64)
        
        bounds = (lb, ub)
        
        # Optimize each block separately
        for block_idx in range(4):
            # Get required capacity for this block
            secret_bits = self.secret_parts[block_idx]
            required_capacity = len(secret_bits)
            
            # Initialize particles with focus on maximizing capacity
            initial_positions = []
            for _ in range(self.n_particles):
                # Try to find a good initial point with sufficient capacity
                for attempt in range(10):  # Try up to 10 times to find good initial positions
                    # Generate random parameters
                    direction = random.randint(0, 15)
                    x_offset = random.randint(0, min(5, max_x))  # Small offset for better capacity
                    y_offset = random.randint(0, min(5, max_y))  # Small offset for better capacity
                    bit_planes = random.randint(0, 15)
                    sb_pole = random.randint(0, 1)
                    sb_dire = random.randint(0, 1)
                    
                    # Check capacity for these parameters
                    capacity = self._calculate_capacity(direction, x_offset, y_offset, self.blocks[block_idx]['data'])
                    
                    # If capacity is sufficient, use these parameters
                    if capacity >= required_capacity:
                        initial_positions.append([direction, x_offset, y_offset, bit_planes, sb_pole, sb_dire])
                        break
                    
                # If we couldn't find parameters with sufficient capacity, use defaults
                if len(initial_positions) <= _ and attempt == 9:
                    # Default to parameters that maximize capacity (typically small offsets)
                    initial_positions.append([0, 0, 0, 1, 0, 0])
            
            # Create optimizer with initial positions if we have any
            options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}
            if len(initial_positions) == self.n_particles:
                # Use our carefully selected initial positions - ENSURE FLOAT DATA TYPE
                optimizer = ps.single.GlobalBestPSO(
                    n_particles=self.n_particles, 
                    dimensions=6, 
                    options=options, 
                    bounds=bounds,
                    init_pos=np.array(initial_positions, dtype=np.float64)  # Explicitly set dtype to float64
                )
            else:
                # Use default initialization
                optimizer = ps.single.GlobalBestPSO(
                    n_particles=self.n_particles, 
                    dimensions=6, 
                    options=options, 
                    bounds=bounds
                )
            
            # Define a lambda function that captures block_idx
            cost_function = lambda particles: self._fitness_function(particles, block_idx)
            
            # Perform optimization
            best_cost, best_pos = optimizer.optimize(cost_function, iters=self.max_iterations,verbose = False)
            
            # Verify final capacity
            direction = int(best_pos[0]) % 16
            x_offset = int(best_pos[1]) % self.block_width
            y_offset = int(best_pos[2]) % self.block_height
            final_capacity = self._calculate_capacity(direction, x_offset, y_offset, self.blocks[block_idx]['data'])
            
            print(f"Block {block_idx+1} optimization complete:")
            print(f"Best fitness: {-best_cost}")
            print(f"Required capacity: {required_capacity}, Final capacity: {final_capacity}")
            
            if final_capacity < required_capacity:
                print(f"WARNING: Best solution for Block {block_idx+1} still has insufficient capacity!")
                # Force parameters that maximize capacity
                # Use a different direction that might provide better coverage
                for test_direction in range(16):
                    test_capacity = self._calculate_capacity(test_direction, 0, 0, self.blocks[block_idx]['data'])
                    if test_capacity >= required_capacity:
                        best_pos[0] = test_direction
                        best_pos[1] = 0  # Minimal x_offset
                        best_pos[2] = 0  # Minimal y_offset
                        print(f"Forced direction {test_direction} with capacity {test_capacity}")
                        break
            
            optimal_params.append(best_pos)
        
        return optimal_params
    
    def embed(self):
        """
        Embed secret message in host image
        
        Returns:
        --------
        numpy.ndarray
            Stego image with embedded secret message
        """
        # Optimize embedding for each block
        optimal_params = self._optimize_embedding()
        
        # Create stego image (start with a copy of the host image)
        stego_image = self.host_image.copy()
        
        # Track total bits embedded and total bit changes
        total_bits = 0
        total_changes = 0
        block_psnrs = []
        block_ssims = []
        
        # Embed in each block using optimal parameters
        for block_idx in range(4):
            block = self.blocks[block_idx]
            secret_bits = self.secret_parts[block_idx]
            params = optimal_params[block_idx]
            
            # Get original block to count bit changes
            original_block = block['data'].copy()
            
            modified_block, psnr, embedding_ratio = self._embed_in_block(block, secret_bits, params)
            block_psnrs.append(psnr)
            
            # Calculate SSIM for this block
            ssim = self._calculate_ssim(original_block, modified_block['data'])
            block_ssims.append(ssim)
            
            # Update stego image with modified block
            y_start, x_start = block['position']
            y_end = y_start + self.block_height
            x_end = x_start + self.block_width
            stego_image[y_start:y_end, x_start:x_end] = modified_block['data']
            
            # Count bit changes
            diff = np.sum(original_block != modified_block['data'])
            total_changes += diff
            total_bits += len(secret_bits)
            
            print(f"Block {block_idx+1} embedded with PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, Embedding ratio: {embedding_ratio*100:.2f}%")
            print(f"Pixel changes in Block {block_idx+1}: {diff} out of {original_block.size} pixels")
        
        # Calculate overall PSNR and SSIM
        overall_psnr = self._calculate_psnr(self.host_image, stego_image)
        overall_ssim = self._calculate_ssim(self.host_image, stego_image)
        
        print(f"Overall PSNR: {overall_psnr:.2f}")
        print(f"Overall SSIM: {overall_ssim:.4f}")
        print(f"Average block PSNR: {sum(block_psnrs)/len(block_psnrs):.2f}")
        print(f"Average block SSIM: {sum(block_ssims)/len(block_ssims):.4f}")
        print(f"Total bit changes: {total_changes} out of {self.host_image.size} pixels ({total_changes/self.host_image.size*100:.2f}%)")
        
        # Save stego image
        cv2.imwrite(self.output_path, stego_image)
        print(f"Stego image saved to {self.output_path}")
        
        return stego_image, overall_psnr, optimal_params

    def extract(self, stego_image, params, message_length):
        """
        Extract hidden message from stego image
        
        Parameters:
        -----------
        stego_image : numpy.ndarray
            Stego image with hidden message
        params : list
            List of parameters used for embedding
        message_length : int
            Length of the original message in characters
        
        Returns:
        --------
        str
            Extracted secret message
        """
        # Calculate expected bits length
        bits_length = message_length * 8
        
        # Split stego image into blocks
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
        
        # Calculate bits per block
        bits_per_block = bits_length // 4
        
        # Extract bits from each block
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
            
            # Get bit planes
            bit_planes = self._get_bit_planes_mask(bit_planes_value)
            
            # Get scanning sequence
            coordinates = self._get_scanning_sequence(direction, x_offset, y_offset, block['data'])
            
            # Extract bits using the scanning sequence
            block_bits = ""
            bits_to_extract = bits_per_block if block_idx < 3 else (bits_length - block_idx * bits_per_block)
            
            for i in range(min(bits_to_extract, len(coordinates))):
                y, x = coordinates[i]
                pixel_value = block['data'][y, x]
                bit_plane = bit_planes[i % len(bit_planes)]
                
                # Extract bit from the specified bit plane
                binary = format(pixel_value, '08b')
                extracted_bit = binary[-(bit_plane+1)]
                block_bits += extracted_bit
            
            # Undo SB-Dire if applied
            if sb_dire == 1:
                block_bits = block_bits[::-1]
            
            # Undo SB-Pole if applied
            if sb_pole == 1:
                block_bits = ''.join('1' if bit == '0' else '0' for bit in block_bits)
            
            extracted_bits += block_bits
        
        # Convert bits to message
        extracted_message = self._bits_to_message(extracted_bits[:bits_length])
        
        return extracted_message


def main():
    # Parameters
    host_image_path = "./images/sail.tiff"
    output_path = "stego_sail.jpg"
    message_file = "./message_20k.txt"
    
    # Read secret message from file
    try:
        with open(message_file, 'r', encoding='utf-8') as file:
            secret_message = file.read()
            print(f"Secret message read from {message_file}")
            print(f"Message length: {len(secret_message)} characters")
    except FileNotFoundError:
        print(f"Error: File {message_file} not found.")
        # Use a default message for testing
        secret_message = "This is a secret message for testing steganography."
        print(f"Using default message: '{secret_message}'")
    except Exception as e:
        print(f"Error reading message file: {str(e)}")
        return
    
    # Create steganography object
    stego = ImageSteganography(host_image_path, secret_message, output_path)
    
    # Embed secret message
    stego_image, psnr, params = stego.embed()
    
    print("Steganography complete!")
    print(f"PSNR: {psnr:.2f}")
    
    # Verify by extracting
    try:
        extracted_message = stego.extract(stego_image, params, len(secret_message))
        print(f"Extracted message: '{extracted_message[:100]}...'")  # Print first 100 chars
        
        # Check if extraction was successful
        if extracted_message == secret_message:
            print("Extraction successful! The messages match.")
        else:
            print("Extraction failed. The messages don't match.")
            
            # Calculate accuracy
            min_len = min(len(secret_message), len(extracted_message))
            correct_chars = sum(1 for a, b in zip(secret_message[:min_len], extracted_message[:min_len]) if a == b)
            accuracy = correct_chars / len(secret_message) * 100 if len(secret_message) > 0 else 0
            print(f"Extraction accuracy: {accuracy:.2f}%")
            
            # Check where differences occur
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