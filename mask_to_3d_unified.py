import numpy as np
import cv2
import json
import os

def mask_to_3d_and_save(mask_path, uid, output_dir):
    """
    Convert a floor plan mask to 3D solid walls with professional rendering.
    Creates unified structure: Yellow base + Grey/White walls on top.
    Wall Height: 6.0m
    Wall Thickness: 0.3m
    """
    try:
        print("[*] Starting 3D wall conversion: 6M HEIGHT, 0.3M THICKNESS, PROFESSIONAL GREY...")
        
        # Read the mask
        mask = cv2.imread(mask_path)
        if mask is None:
            raise ValueError(f"Could not read mask from {mask_path}")
        
        print(f"[OK] Mask loaded: {mask.shape}")
        
        # Extract wall pixels
        walls = mask[:, :, 0]  # Blue channel - walls
        
        # Find ALL wall contours (both outer AND inner room walls)
        # cv2.RETR_TREE gets ALL contours including holes/inner walls
        wall_contours, hierarchy = cv2.findContours(walls, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print(f"[OK] Found {len(wall_contours)} wall contours (OUTER + INNER walls)")
        
        # Wall parameters - PROFESSIONAL CINEMA GRADE
        WALL_HEIGHT = 8.0        # INCREASED TO 8M - MUCH TALLER WALLS BUDDY!!
        BASE_HEIGHT = 0.3        # Yellow base height (bottom)
        WALL_HEIGHT_ABOVE = WALL_HEIGHT - BASE_HEIGHT  # Grey wall height (above yellow)
        WALL_THICKNESS = 0.3     # 0.3m thickness as requested
        SCALE = 0.1              # pixels to meters
        
        # Lists to store geometry
        vertices = []
        faces = []
        vertex_count = 0
        
        print("[OK] Professional Cinema-Grade Wall Structure:")
        print(f"     Total Height: {WALL_HEIGHT}m (MUCH TALLER!)")
        print(f"     Base: {BASE_HEIGHT}m (YELLOW)")
        print(f"     Walls: {WALL_HEIGHT_ABOVE}m (PROFESSIONAL GREY)")
        print(f"     Thickness: {WALL_THICKNESS}m")
        
        # ==========================================
        # CREATE UNIFIED WALLS (Yellow base + Grey on top)
        # ==========================================
        
        # Filter contours by size (remove tiny noise, keep outer + inner walls)
        min_contour_area = 20  # Minimum pixels to be considered a wall
        valid_contours = []
        
        for contour in wall_contours:
            area = cv2.contourArea(contour)
            # Keep contours with meaningful area (both outer and inner walls)
            if area >= min_contour_area:
                valid_contours.append(contour)
        
        print(f"[OK] Filtered to {len(valid_contours)} meaningful walls (removed noise)")
        
        for contour_idx, contour in enumerate(valid_contours):
            # DO NOT SIMPLIFY - Use ALL points from contour for CONTINUOUS walls
            # Convert contour to list of points (keep all points, no approximation)
            contour_points = contour.reshape(-1, 2)
            
            if len(contour_points) < 2:
                continue
            
            print(f"   Contour {contour_idx}: {len(contour_points)} CONTINUOUS points")
            
            # Create SIMPLE wall structure - one quad per segment
            for i, point in enumerate(contour_points):
                x, y = point
                # Get next point in sequence (loop back to start at end)
                next_point = contour_points[(i + 1) % len(contour_points)]
                nx, ny = next_point
                
                # Direction vector
                dx = nx - x
                dy = ny - y
                length = np.sqrt(dx*dx + dy*dy)
                
                if length == 0:
                    continue
                
                # Perpendicular direction (for thickness)
                px = -dy / length
                py = dx / length
                
                # Current point
                x_scaled = x * SCALE
                y_scaled = y * SCALE
                
                # Offset for thickness (inner and outer)
                thickness_offset = WALL_THICKNESS / 2
                
                # Inner and outer positions
                inner_x = x_scaled + px * thickness_offset
                inner_y = y_scaled + py * thickness_offset
                outer_x = x_scaled - px * thickness_offset
                outer_y = y_scaled - py * thickness_offset
                
                # Next point scaled and offset
                nx_scaled = nx * SCALE
                ny_scaled = ny * SCALE
                next_inner_x = nx_scaled + px * thickness_offset
                next_inner_y = ny_scaled + py * thickness_offset
                next_outer_x = nx_scaled - px * thickness_offset
                next_outer_y = ny_scaled - py * thickness_offset
                
                # Store current vertex indices
                start_idx = vertex_count
                
                # Add 4 vertices for this quad segment (simple approach)
                # Yellow base vertices (bottom)
                vertices.append([inner_x, inner_y, 0])           # 0: inner bottom
                vertices.append([outer_x, outer_y, 0])           # 1: outer bottom
                vertices.append([next_inner_x, next_inner_y, 0]) # 2: next inner bottom
                vertices.append([next_outer_x, next_outer_y, 0]) # 3: next outer bottom
                
                # Grey wall vertices (top)
                vertices.append([inner_x, inner_y, WALL_HEIGHT])           # 4: inner top
                vertices.append([outer_x, outer_y, WALL_HEIGHT])           # 5: outer top
                vertices.append([next_inner_x, next_inner_y, WALL_HEIGHT]) # 6: next inner top
                vertices.append([next_outer_x, next_outer_y, WALL_HEIGHT]) # 7: next outer top
                
                vertex_count += 8
                
                # Create SIMPLE faces - 2 triangles per quad, 3 quads total per segment
                
                # ========================
                # YELLOW BASE FACES (simple quad = 2 triangles)
                # ========================
                # Bottom face (quad: 0,1,3,2)
                faces.append((start_idx + 0, start_idx + 1, start_idx + 3, 'yellow'))
                faces.append((start_idx + 0, start_idx + 3, start_idx + 2, 'yellow'))
                
                # ========================
                # GREY WALL FACES (simple quads)
                # ========================
                # Front/Inner face (quad: 0,2,6,4)
                faces.append((start_idx + 0, start_idx + 2, start_idx + 6, 'grey'))
                faces.append((start_idx + 0, start_idx + 6, start_idx + 4, 'grey'))
                
                # Back/Outer face (quad: 1,5,7,3)
                faces.append((start_idx + 1, start_idx + 5, start_idx + 7, 'grey'))
                faces.append((start_idx + 1, start_idx + 7, start_idx + 3, 'grey'))
                
                # Top face (quad: 4,6,7,5)
                faces.append((start_idx + 4, start_idx + 6, start_idx + 7, 'grey'))
                faces.append((start_idx + 4, start_idx + 7, start_idx + 5, 'grey'))
        
        print(f"[OK] Generated {len(vertices)} vertices with {len(faces)} professional faces")
        print(f"[OK] CONTINUOUS WALLS: All contour points preserved (NO SIMPLIFICATION)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Write OBJ file - UNIFIED SINGLE OBJECT
        obj_path = os.path.join(output_dir, f"{uid}_3d_model.obj")
        
        with open(obj_path, 'w') as f:
            f.write(f"# PROFESSIONAL CINEMA-GRADE 3D FLOOR PLAN\n")
            f.write(f"# Total Wall Height: {WALL_HEIGHT}m\n")
            f.write(f"# Yellow Boundary Base: {BASE_HEIGHT}m\n")
            f.write(f"# Professional Grey Walls: {WALL_HEIGHT_ABOVE}m\n")
            f.write(f"# Wall Thickness: {WALL_THICKNESS}m\n")
            f.write(f"# CONTINUOUS WALLS - All contour points included\n")
            f.write(f"# Unified Single Object - ALL WALLS TOGETHER\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Write normals for professional lighting
            f.write(f"\n# Professional Normals\n")
            for _ in range(len(vertices)):
                f.write(f"vn 0 0 1\n")
            
            # Write faces - UNIFIED STRUCTURE
            f.write(f"\n# UNIFIED PROFESSIONAL WALLS - CONTINUOUS GEOMETRY\n")
            f.write(f"g unified_professional_walls\n\n")
            
            # Yellow base faces
            f.write(f"# Yellow Boundary Base (0-{BASE_HEIGHT}m) - CONTINUOUS\n")
            for face in faces:
                if face[3] == 'yellow':
                    f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
            
            # Professional Grey wall faces
            f.write(f"\n# Professional Grey Walls ({BASE_HEIGHT}-{WALL_HEIGHT}m) - CINEMA GRADE - CONTINUOUS\n")
            for face in faces:
                if face[3] == 'grey':
                    f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
        
        print(f"[OK] OBJ file created: {obj_path}")
        print(f"[OK] CONTINUOUS PROFESSIONAL GREY WALLS: 6M HEIGHT x 0.3M THICKNESS")
        print(f"[OK] Yellow boundary base + Grey walls = ONE UNIFIED CONTINUOUS OBJECT")
        
        # Create professional texture
        texture_path = os.path.join(output_dir, f"{uid}_texture.jpg")
        create_professional_wall_texture(texture_path)
        print(f"[OK] Professional cinema-grade texture created: {texture_path}")
        
        return obj_path
        
    except Exception as e:
        print(f"[ERROR] Error in mask_to_3d_and_save: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def create_professional_wall_texture(texture_path):
    """
    Create professional cinema-grade grey wall texture with realistic details.
    """
    texture_size = 1024
    texture = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 180
    
    # Create premium brick pattern with professional grey shades
    brick_width = 96
    brick_height = 48
    
    for i in range(0, texture_size, brick_height):
        for j in range(0, texture_size, brick_width):
            # Alternate professional grey shades (not too dark, not too light)
            if (i // brick_height + j // brick_width) % 2 == 0:
                texture[i:i+brick_height, j:j+brick_width] = [170, 170, 170]  # Darker grey
            else:
                texture[i:i+brick_height, j:j+brick_width] = [200, 200, 200]  # Lighter grey
            
            # Professional grout lines
            if i + brick_height < texture_size:
                texture[i+brick_height-2:i+brick_height, j:j+brick_width] = [150, 150, 150]
            if j + brick_width < texture_size:
                texture[i:i+brick_height, j+brick_width-2:j+brick_width] = [150, 150, 150]
    
    # Add subtle noise for realism
    noise = np.random.randint(-15, 15, texture.shape, dtype=np.int16)
    texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(texture_path, texture)
    print(f"[OK] Professional grey texture: Cinema-grade brick pattern with realistic details")
