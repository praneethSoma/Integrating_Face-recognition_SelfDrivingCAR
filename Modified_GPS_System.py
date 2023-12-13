import pygame
import numpy as np
import csv

def save_spawn_and_destination(spawn_term, spawn, dest_term, dest):
    with open('selected_spawn_and_destination.txt', 'w') as file:
        file.write(f"{spawn_term},{spawn[0]},{spawn[1]}\n")
        file.write(f"{dest_term},{dest[0]},{dest[1]}\n")
        
def load_spawn_points(filename):
    spawn_points = []
    spawn_terms = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            spawn_points.append((float(row[1]), float(row[2])))
            spawn_terms.append(row[0])
    return spawn_points, spawn_terms


# Convert pixel coordinates to CARLA world coordinates using the transformation
def pixel_to_carla(pixel_x, pixel_y):
    est_carla_x = 0.02375 * pixel_x - 0.39100 * pixel_y - 3.6505
    est_carla_y = 0.26673 * pixel_x + 0.02252 * pixel_y + 43.8146
    return est_carla_x, est_carla_y

# Refined function to convert CARLA world coordinates back to pixel coordinates for displaying on the image
def carla_to_pixel(carla_x, carla_y, image_width, image_height):
    pixel_x = image_width // 2 + carla_x
    pixel_y = image_height // 2 - carla_y  # Inverting y-axis since in image coordinates, top is 0
    return int(pixel_x), int(pixel_y)

# Function to find the nearest spawn point and its term
def find_nearest_spawn_point_and_term(x, y, spawn_points, spawn_terms):
    distances = [np.sqrt((x - sx)**2 + (y - sy)**2) for sx, sy in spawn_points]
    nearest_index = np.argmin(distances)
    return spawn_terms[nearest_index], spawn_points[nearest_index]

# Load spawn points and their terms
spawn_points, spawn_terms = load_spawn_points('spawn_points.csv')

def select_spawn_and_destination():
    pygame.init()

    # Load the image
    image = pygame.image.load('Town01.jpg')
    screen = pygame.display.set_mode(image.get_size())
    pygame.display.set_caption('Map Destination Selector')
    
    spawn_term, spawn, dest_term, dest = None, None, None, None 
    
    running = True
    first_click = True  # Flag to track if it's the first click (for spawn point) or second (for destination)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return None, None, None, None
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get the mouse position
                user_pixel_x, user_pixel_y = pygame.mouse.get_pos()
                
                # Convert to CARLA coordinates
                user_carla_x, user_carla_y = pixel_to_carla(user_pixel_x - image.get_width() // 2, user_pixel_y - image.get_height() // 2)
                
                # Find the nearest spawn point and its term
                nearest_term, nearest_spawn = find_nearest_spawn_point_and_term(user_carla_x, user_carla_y, spawn_points, spawn_terms)
                
                if first_click:  # If it's the first click, set the spawn point
                    spawn_term, spawn = nearest_term, nearest_spawn
                    pygame.draw.circle(image, (0, 255, 0), (user_pixel_x, user_pixel_y), 5)  # Green circle for the spawn point
                    first_click = False
                else:  # If it's the second click, set the destination point
                    dest_term, dest = nearest_term, nearest_spawn
                    pygame.draw.circle(image, (255, 0, 0), (user_pixel_x, user_pixel_y), 5)  # Red circle for the destination point
                    save_spawn_and_destination(spawn_term, spawn, dest_term, dest)
                    #print(f"Chosen CARLA spawn point: {spawn_term} ({spawn})")
                    #print(f"Chosen CARLA destination: {dest_term} ({dest})")
                    pygame.quit()
                    return spawn_term, spawn, dest_term, dest
            
        screen.blit(image, (0, 0))
        pygame.display.flip()



    
#The code below can be used to run the GPS module 
if __name__ == "__main__":
    select_spawn_and_destination()