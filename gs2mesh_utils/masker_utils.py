# =============================================================================
#  Imports
# =============================================================================

import numpy as np
import os
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

from gs2mesh_utils.transformation_utils import project_depth_image

# =============================================================================
#  Helper functions
# =============================================================================

def init_predictor(base_dir, device='cuda'):
    """
    Initialize the SAM predictor.

    Parameters:
    base_dir (str): Base directory of the repository.
    device (str): Device to run the model on.

    Returns:
    SamPredictor: Initialized SAM predictor.
    """
    sam_checkpoint = os.path.join(base_dir, 'third_party', 'SAM', 'sam_vit_h_4b8939.pth')
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def farthest_point_sampling(points, num_seeds):
    """
    Given an array of 2D points and an initial random seed, sample additional points using farthest point sampling (FPS).

    Parameters:
    points (np.ndarray): Array of 2D points.
    num_seeds (int): Number of points to sample.

    Returns:
    np.ndarray: Array of sampled points.
    """
    farthest_pts = np.zeros((num_seeds, 2), dtype=np.float32)
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.full(len(points), np.inf)

    for i in range(1, num_seeds):
        dist = np.sum((points - farthest_pts[i-1])**2, axis=1)
        distances = np.minimum(distances, dist)
        farthest_pts[i] = points[np.argmax(distances)]

    return farthest_pts

# =============================================================================
#  Class for SAM Masker
# =============================================================================

class Masker:
    def __init__(self, predictor, renderer, stereo, image_number=0):
        """
        Initialize the Masker class.

        Parameters:
        predictor (SamPredictor): SAM predictor.
        renderer (Renderer): Renderer class object.
        stereo (Stereo): Stereo class object.
        image_number (int): Base image number from which to start masking.
        """
        self.predictor = predictor        
        self.index = image_number
        self.renderer = renderer
        self.stereo = stereo
        self.image = np.array(Image.open(os.path.join(renderer.render_folder_name(self.index), 'left.png')))

        self.points = []
        self.bboxes = None
        self.mask = None

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.dragging = False
        self.drag_start = None
        self.drag_threshold = 5
        
        self.display_image()
    
    def display_image(self):
        """
        Display the current image being masked along with points, mask, and bounding box.

        Returns:
        None
        """
        self.ax.clear()
        img = self.image
        self.ax.imshow(img)
        self.ax.set_title(f'Image {self.index:03}')

        if self.mask is not None:
            self.show_mask(self.mask, self.ax)
        if len(self.points) > 0:
            self.show_points(np.array([point[0] for point in self.points]), np.array([point[1] for point in self.points]), self.ax)
        if self.bboxes is not None:
            self.show_box(self.bboxes, self.ax)
        self.fig.canvas.draw()

    def on_click(self, event):
        """
        Handle mouse click events for adding points or starting a bounding box.

        Parameters:
        event (MouseEvent): Matplotlib mouse event.

        Returns:
        None
        """
        if event.inaxes != self.ax:
            return
        self.drag_start = (event.xdata, event.ydata)
        self.dragging = True # Assume dragging starts, will verify in on_release

    def on_release(self, event):
        """
        Handle mouse release events for adding points or finishing a bounding box.

        Parameters:
        event (MouseEvent): Matplotlib mouse event.

        Returns:
        None
        """
        if self.dragging and event.inaxes == self.ax:
            drag_end = (event.xdata, event.ydata)
            dist_moved = np.sqrt((drag_end[0] - self.drag_start[0])**2 + (drag_end[1] - self.drag_start[1])**2)
            
            if dist_moved < self.drag_threshold: # Treat as a click
                if event.button == 1: # Left-click
                    label = 1 
                    self.points.append((self.drag_start, label))
                elif event.button == 3: # Right-click
                    label = 0
                    self.points.append((self.drag_start, label))
                elif event.button == 2: # Middle-click
                    if not self.remove_bbox_if_near(self.drag_start[0], self.drag_start[1]):
                        self.remove_point(self.drag_start[0], self.drag_start[1])
            else: # Treat as a drag
                if event.button == 1:
                    self.bboxes = np.array([self.drag_start[0], self.drag_start[1], drag_end[0], drag_end[1]])

            self.dragging = False
            self.redraw()
            
    def remove_bbox_if_near(self, x, y, threshold=20):
        """
        Remove the bounding box if the middle-click is near it.

        Parameters:
        x (float): X coordinate of middle-click.
        y (float): Y coordinate of middle-click.
        threshold (float): Distance threshold to consider the middle-click as near.

        Returns:
        bool: True if bounding box was removed, False otherwise.
        """
        if self.bboxes is None:
            return False
        bbox = self.bboxes
        near_left_or_right = min(abs(x - bbox[0]), abs(x - bbox[2])) < threshold
        near_top_or_bottom = min(abs(y - bbox[1]), abs(y - bbox[3])) < threshold
        if near_left_or_right or near_top_or_bottom:
            self.bboxes = None
            return True
        return False
    
    def remove_point(self, x, y):
        """
        Remove the point nearest to (x, y).

        Parameters:
        x (float): X coordinate.
        y (float): Y coordinate.

        Returns:
        None
        """
        if not self.points:
            return
        nearest_point_index = min(range(len(self.points)), key=lambda i: (self.points[i][0][0] - x) ** 2 + (self.points[i][0][1] - y) ** 2)
        self.points.pop(nearest_point_index)

    def redraw(self):
        """
        Redraw the image with the updated points, mask, and bounding box.

        Returns:
        None
        """
        if len(self.points) > 0 or self.bboxes is not None:
            self.predictor.set_image(self.image)
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array([point[0] for point in self.points]) if len(self.points) > 0 else None,
                point_labels=np.array([point[1] for point in self.points]) if len(self.points) > 0 else None,
                box=self.bboxes if self.bboxes is not None else None,
                multimask_output=True,
            )
            self.mask = masks[2, :, :]
            output_dir = self.renderer.render_folder_name(self.index)
            np.save(os.path.join(output_dir, 'left_mask.npy'), self.mask)
            plt.imsave(os.path.join(output_dir, 'left_mask.png'), self.mask)
        else:
            self.mask = np.zeros_like(self.image)[:, :, 0].astype(bool)
        self.display_image()

    def show_mask(self, mask, ax):
        """
        Show the mask on the image.

        Parameters:
        mask (np.ndarray): Mask to display.
        ax (Axes): Matplotlib axes.

        Returns:
        None
        """
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self, coords, labels, ax, marker_size=375):
        """
        Show the clicked points on the image.

        Parameters:
        coords (np.ndarray): Array of clicked point coordinates.
        labels (np.ndarray): Array of clicked point labels.
        ax (Axes): Matplotlib axes.
        marker_size (int): Size of the markers to be shown.

        Returns:
        None
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(self, box, ax):
        """
        Show the bounding box on the image.

        Parameters:
        box (np.ndarray): Array of bounding box coordinates [left, top, right, bottom].
        ax (Axes): Matplotlib axes.

        Returns:
        None
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    def segment(self, num_seeds=3, erosion_kernel_size=50, closing_kernel_size=10, resume=0, visualize=False):
        """
        Perform segmentation on a sequence of images using SAM and depth projections.

        Parameters:
        num_seeds (int): Number of points to sample from inside/outside the mask using farthest point sampling.
        erosion_kernel_size (int): Kernel size for erosion filter on mask to avoid leaking from the object to background.
        closing_kernel_size (int): Kernel size for closing to close any holes that might have formed with the erosion.
        resume (int): Index of view from which to start masking
        visualize (bool): Flag to visualize the segmentation process for debugging.

        Returns:
        None
        """
        prev_mask = None
        mask = None
        point_coords = None
        point_labels = None
        bbox = None
        prev_image = None
        for camera_number in tqdm(range(resume, len(self.renderer))):
            output_dir = os.path.join(self.renderer.render_folder_name(camera_number))
            image = np.array(Image.open(os.path.join(output_dir, 'left.png'))).astype(np.uint8)
            depth = np.load(os.path.join(output_dir, f'out_{self.stereo.model_name}', 'depth.npy'))

            if camera_number == resume:
                mask = np.load(os.path.join(output_dir, 'left_mask.npy'))
            elif camera_number > resume:
                mask0 = prev_mask
                im0 = prev_image
                im1 = image
                d0 = np.load(os.path.join(self.renderer.render_folder_name(camera_number-1), f'out_{self.stereo.model_name}', 'depth.npy'))
                K0 = self.renderer.left_cameras[camera_number-1]['intrinsic']
                ext0 = self.renderer.left_cameras[camera_number-1]['extrinsic']
                K1 = self.renderer.left_cameras[camera_number]['intrinsic']
                ext1 = self.renderer.left_cameras[camera_number]['extrinsic']
                projected_mask = project_depth_image(d0, mask0, K1, K0, ext1[:3, :3], ext1[:3, 3], ext0[:3, :3], ext0[:3, 3]) > 0.5
                projected_negative_mask = project_depth_image(d0, ~mask0, K1, K0, ext1[:3, :3], ext1[:3, 3], ext0[:3, :3], ext0[:3, 3]) > 0.5
                
                closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
                erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
                closing = cv2.morphologyEx(projected_mask.astype(np.uint8), cv2.MORPH_CLOSE, closing_kernel)
                erosion = cv2.erode(closing, erosion_kernel, iterations=1)
                projected_mask = erosion > 0.5

                closing = cv2.morphologyEx(projected_negative_mask.astype(np.uint8), cv2.MORPH_CLOSE, closing_kernel)
                erosion = cv2.erode(closing, erosion_kernel, iterations=1)
                projected_negative_mask = erosion > 0.5

                positive_true_indices = np.argwhere(projected_mask)
                positive_point_coords = farthest_point_sampling(positive_true_indices.astype(np.float32), num_seeds)
                positive_point_coords = positive_point_coords[:, [1, 0]]
                positive_point_labels = np.ones(positive_point_coords.shape[0])

                negative_true_indices = np.argwhere(projected_negative_mask)
                negative_point_coords = farthest_point_sampling(negative_true_indices.astype(np.float32), num_seeds)
                negative_point_coords = negative_point_coords[:, [1, 0]]
                negative_point_labels = np.zeros(negative_point_coords.shape[0])

                point_coords = np.concatenate([positive_point_coords, negative_point_coords], axis=0)
                point_labels = np.concatenate([positive_point_labels, negative_point_labels], axis=0)
                
                self.predictor.set_image(image)
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=bbox,
                    multimask_output=True,
                )
                mask = masks[2, :, :]
                    
            if visualize:
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                self.show_mask(mask, plt.gca())
                if point_coords is not None:
                    self.show_points(point_coords, point_labels, plt.gca())
                if bbox is not None:
                    self.show_box(bbox, plt.gca())
                plt.title(f"Image #{camera_number}", fontsize=18)
                plt.show() 
        
            np.save(os.path.join(output_dir, 'left_mask.npy'), mask)
            plt.imsave(os.path.join(output_dir, 'left_mask.png'), mask)
            prev_mask = mask
            prev_image = image
