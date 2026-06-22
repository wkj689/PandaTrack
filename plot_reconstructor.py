import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import datetime as dt
from pathlib import Path
import os
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class PlotReconstructor:
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.trajectory_db = pd.DataFrame()
        self.grid_data = {
            "dwell_time": np.zeros((0, 0)),
            "activity_time": np.zeros((0, 0))
        }
        self.grid_rows = 0
        self.grid_cols = 0
        self.video_fps = 30
        self.frame_skip = 2
        self.video_width = 0
        self.video_height = 0

    def load_database(self):
        """Load trajectory database Excel file"""
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database file does not exist: {self.database_path}")

        # Read all sheets
        xls = pd.ExcelFile(self.database_path)
        sheet_names = xls.sheet_names

        # Check for a trajectory data sheet
        if 'TrajectoryEN' in sheet_names:
            self.trajectory_db = pd.read_excel(self.database_path, sheet_name='TrajectoryEN')
        else:
            # If not found, try the first sheet
            self.trajectory_db = pd.read_excel(self.database_path, sheet_name=0)

        # Ensure datetime column is datetime dtype
        self.trajectory_db['datetime'] = pd.to_datetime(self.trajectory_db['datetime'])

        # Get basic video information
        if not self.trajectory_db.empty:
            self.video_width = self.trajectory_db['frame_width'].iloc[0]
            self.video_height = self.trajectory_db['frame_height'].iloc[0]
            self.video_fps = self.trajectory_db['fps'].iloc[0]
            self.frame_skip = self.trajectory_db['frame_skip'].iloc[0]
            self.grid_rows = self.trajectory_db['grid_rows'].iloc[0]
            self.grid_cols = self.trajectory_db['grid_cols'].iloc[0]

            # Initialize grid data
            self.grid_data = {
                "dwell_time": np.zeros((self.grid_rows, self.grid_cols)),
                "activity_time": np.zeros((self.grid_rows, self.grid_cols))
            }

        return self.trajectory_db

    def filter_by_time(self, start_time: dt.datetime, end_time: dt.datetime):
        """Filter data by time range"""
        if self.trajectory_db.empty:
            return pd.DataFrame()

        mask = (self.trajectory_db['datetime'] >= start_time) & (self.trajectory_db['datetime'] <= end_time)
        return self.trajectory_db.loc[mask]

    def reconstruct_heatmap(self, filtered_df: pd.DataFrame, output_path: Path):
        """Reconstruct heatmap from filtered data"""
        if filtered_df.empty:
            print("No data; cannot generate heatmap")
            return

        # Reset grid data
        self.grid_data["dwell_time"] = np.zeros((self.grid_rows, self.grid_cols))
        self.grid_data["activity_time"] = np.zeros((self.grid_rows, self.grid_cols))

        # Compute grid cell size
        grid_cell_w = self.video_width / self.grid_cols
        grid_cell_h = self.video_height / self.grid_rows

        # Iterate data and accumulate dwell/activity time
        for _, row in filtered_df.iterrows():
            if pd.isna(row['grid_row']) or pd.isna(row['grid_col']):
                continue

            grid_row = int(row['grid_row'])
            grid_col = int(row['grid_col'])

            # Compute time increment (s)
            delta_t = 1 / self.video_fps * self.frame_skip

            if row['state'] == 'static':
                self.grid_data["dwell_time"][grid_row, grid_col] += delta_t
            elif row['state'] == 'moving':
                self.grid_data["activity_time"][grid_row, grid_col] += delta_t

        # Create blank canvas
        canvas = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)

        # Compute max dwell time
        max_dwell = np.max(self.grid_data["dwell_time"])
        if max_dwell == 0:
            max_dwell = 1  # Avoid division by zero

        # DwellTime
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                dwell_time = self.grid_data["dwell_time"][i, j]
                if dwell_time > 0:
                    #  (0-255)
                    intensity = int(255 * (dwell_time / max_dwell))
                    color = (0, 0, intensity)  # BGR

                    
                    x1 = int(j * grid_cell_w)
                    y1 = int(i * grid_cell_h)
                    x2 = int((j + 1) * grid_cell_w)
                    y2 = int((i + 1) * grid_cell_h)

                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)

                    # DwellTime
                    dwell_min = dwell_time / 60
                    text = f"{dwell_min:.1f}m"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = x1 + (grid_cell_w - text_size[0]) // 2
                    text_y = y1 + (grid_cell_h + text_size[1]) // 2

                    cv2.putText(canvas, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        cv2.imwrite(str(output_path), canvas)
        print(f"HeatmapEN: {output_path}")

        # Heatmap
        self._generate_raw_heatmap(output_path)

    def _generate_raw_heatmap(self, path: Path):
        """Generate clean pseudocolor heatmap"""
        if np.max(self.grid_data["dwell_time"]) == 0:
            return

        raw_path = path.with_name(path.stem + "_raw.png")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        im = ax.imshow(self.grid_data["dwell_time"], cmap="jet", origin="upper")
        ax.axis("off")

        
        cb_ax = inset_axes(
            ax,
            width="3%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(im, cax=cb_ax)

        def _fmt(v, _):
            sec = v
            return f"{sec / 60:.1f}"

        cbar.formatter = ticker.FuncFormatter(_fmt)
        cbar.set_label("Dwell Time")
        cbar.update_ticks()

        fig.savefig(str(raw_path), transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"Clean heatmap saved: {raw_path}")

    def reconstruct_trajectory(self, filtered_df: pd.DataFrame, output_path: Path, background_path: str = None):
        """Reconstruct trajectory plot from filtered data"""
        if filtered_df.empty:
            print("No data; cannot generate trajectory plot")
            return

        # Create background
        if background_path and Path(background_path).exists():
            bg = cv2.imread(background_path)
            if bg.shape[:2] != (self.video_height, self.video_width):
                bg = cv2.resize(bg, (self.video_width, self.video_height))
        else:
            # Create blank background
            bg = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
            bg[:] = (50, 50, 50)  # Dark gray background

        # Sort by time
        filtered_df = filtered_df.sort_values('datetime')

        # Draw trajectory
        prev_point = None
        for _, row in filtered_df.iterrows():
            if pd.isna(row['x']) or pd.isna(row['y']):
                prev_point = None
                continue

            x, y = int(row['x']), int(row['y'])
            state = row['state']

            # Choose color by state
            if state == 'moving':
                color = (0, 0, 255)  # Red - moving
                radius = 3
            elif state == 'static':
                color = (0, 255, 0)  # Green - static
                radius = 2
            else:  # absent
                prev_point = None
                continue

            
            cv2.circle(bg, (x, y), radius, color, -1)

            # ()
            if prev_point:
                cv2.line(bg, prev_point, (x, y), (0, 255, 255), 1)  # Yellow connection line

            prev_point = (x, y)

        # Add time-range text
        start_str = filtered_df['datetime'].min().strftime("%Y-%m-%d %H:%M")
        end_str = filtered_df['datetime'].max().strftime("%Y-%m-%d %H:%M")
        text = f"{start_str} EN {end_str}"
        cv2.putText(bg, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        cv2.imwrite(str(output_path), bg)
        print(f"Trajectory plot savedEN: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ENTrajectoryENHeatmapENTrajectory")
    parser.add_argument("--database", required=True, help="TrajectoryENExcelEN")
    parser.add_argument("--start", help="EN,EN:YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end", help="EN,EN:YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--heatmap", help="ENHeatmapENPath")
    parser.add_argument("--trajectory", help="ENTrajectoryENPath")
    parser.add_argument("--background", help="TrajectoryENPath(EN)")
    args = parser.parse_args()

    reconstructor = PlotReconstructor(args.database)
    reconstructor.load_database()

    
    if args.start and args.end:
        start_time = dt.datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S')
        end_time = dt.datetime.strptime(args.end, '%Y-%m-%d %H:%M:%S')
        filtered_df = reconstructor.filter_by_time(start_time, end_time)
    else:
        print("EN,EN")
        filtered_df = reconstructor.trajectory_db

    # Heatmap
    if args.heatmap:
        reconstructor.reconstruct_heatmap(filtered_df, Path(args.heatmap))

    # Trajectory
    if args.trajectory:
        reconstructor.reconstruct_trajectory(filtered_df, Path(args.trajectory), args.background)


if __name__ == "__main__":
    main()