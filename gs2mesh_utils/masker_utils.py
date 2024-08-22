# =============================================================================
#  Imports
# =============================================================================

import numpy as np
import os
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from gs2mesh_utils.transformation_utils import project_depth_image

# =============================================================================
#  Helper functions
# =============================================================================

def create_temp_jpg_folder(renderer):
    """
    Create folder of jpg images containing the rendered left image from each view.

    Parameters:
    renderer (Renderer): Renderer object class.

    Returns:
    str: The path to the resulting folder.
    """
    dst_dir = os.path.join(renderer.output_dir_root, 'images_jpg')
    os.makedirs(dst_dir, exist_ok=True)
    for camera_number in tqdm(range(len(renderer))):
        new_filename = os.path.join(dst_dir, f"{camera_number:04}.jpg")
        if os.path.exists(new_filename):
            continue
        output_dir = os.path.join(renderer.render_folder_name(camera_number))
        image = Image.open(os.path.join(output_dir, 'left.png'))
        image.convert('RGB').save(new_filename)
    return dst_dir
        
def init_predictor(base_dir, renderer, use_local=False, device='cuda'):
    """
    Initialize the SAM predictor.

    Parameters:
    base_dir (str): Base directory of the repository.
    renderer (Renderer): Renderer object class.
    use_local (bool): Flag to determine whether to use local weights or huggingface weights.
    device (str): Device to run the model on.

    Returns:
    SamPredictor: Initialized SAM predictor.
    """
    predictor = None
    if use_local:
        SAM2_dir = os.path.abspath(os.path.join(base_dir, 'third_party', 'segment-anything-2'))
        checkpoint = os.path.join(SAM2_dir, 'checkpoints', 'sam2_hiera_large.pt')
        model_cfg = 'sam2_hiera_l.yaml'
        cfg_path = os.path.join(SAM2_dir, 'sam2_configs')
        predictor = build_sam2_video_predictor(model_cfg, config_path=cfg_path, ckpt_path=checkpoint, device=device)
    else:
        predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device=device)
    images_dir = create_temp_jpg_folder(renderer)
    inference_state = predictor.init_state(video_path=images_dir)
    return predictor, inference_state, images_dir

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
    def __init__(self, predictor, inference_state, images_dir, renderer, stereo, image_number=0):
        """
        Initialize the Masker class.

        Parameters:
        predictor (SAM2VideoPredictor): SAM2 predictor.
        predictor (dict): SAM2 inference state.
        images_dir (str): directory of jpg images for SAM2.
        renderer (Renderer): Renderer class object.
        stereo (Stereo): Stereo class object.
        image_number (int): Base image number from which to start masking.
        """

        self.predictor = predictor
        self.inference_state = inference_state
        self.predictor.reset_state(self.inference_state)
        
        self.index = image_number
        self.renderer = renderer
        self.stereo = stereo

        image_filenames = [
            p for p in os.listdir(images_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        image_filenames.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        self.image = np.array(Image.open(os.path.join(self.renderer.output_dir_root, 'images_jpg', image_filenames[self.index])))
        
        self.points = []
        self.bboxes = None
        self.mask = None

        plt.close('all')
        
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        plt.subplots_adjust(bottom=0.2)
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.dragging = False
        self.drag_start = None
        self.drag_threshold = 5
        
        self.display_image()
        
    # def __del__(self):
    #     """
    #     Destructor to ensure the figure is closed when the object is destroyed.
    #     """
    #     plt.close('all')
        
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
            self.predictor.reset_state(self.inference_state)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.index,
                obj_id=1,
                points=np.array([point[0] for point in self.points]) if len(self.points) > 0 else None,
                labels=np.array([point[1] for point in self.points]) if len(self.points) > 0 else None,
            )
            self.mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze(0)
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
    
    def segment(self):
        """
        Propagate the mask throughout the video using SAM2 and save the resulting masks.

        Returns:
        None
        """
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze(0)
            output_dir = self.renderer.render_folder_name(out_frame_idx)
            np.save(os.path.join(output_dir, 'left_mask.npy'), mask)
            plt.imsave(os.path.join(output_dir, 'left_mask.png'), mask)
        plt.close('all')
