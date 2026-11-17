## Introduction

This module is designed to enable direct communication between ROS 2 and 3D Slicer.

The documentation is available on readthedocs:
https://slicer-ros2.readthedocs.io/.  Make sure you
use the correct version (drop-down menu under the SlicerROS2 logo).

Please check out this video below for an overview of some of the features and use-cases.

<div align="center">
  <a href="https://www.youtube.com/watch?v=k3VGs0YaE3g">
    <img src="https://img.youtube.com/vi/k3VGs0YaE3g/maxresdefault.jpg" alt="SlicerROS2 demonstration" style="max-width:100%; height:auto;">
  </a>
</div>

## Motion Control
**Steps to facilitate motion planning and control of the Micromate robot based on a generated trajectory.**
[edit script paths]
Setup:

1. Build Slicer using the [generic build instructions](https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/linux.html).

2. Compile the SlicerROS2 module according to [these instructions](https://slicer-ros2.readthedocs.io/en/v1.2/pages/getting-started.html), using the [my-motion-control branch](https://github.com/msong33/slicer_ros2_module/tree/my-motion-control) of this repository (this and the following steps assume the use of the ROS workspace directory ~/ros2_ws, with the SlicerROS2 module source code cloned under ~/ros2_ws/src/slicer_ros2_module).

3. Clone the [Micromate repository](https://github.com/msong33/micromate) for Micromate-ROS integration into ~/ros2_ws/src/micromate and rebuild the workspace.

4. Download the [VerSe 2020 dataset](https://osf.io/4skx2/files/osfstorage) into ~/ros2_ws/src/dataset for use with the motion control module.

\
Run each of the following commands in separate, sourced terminals to setup the Slicer environment (```cd ros2_ws```, ```source install/setup.bash``` in each terminal):

To load the Micromate robot, run the two following commands in two separate terminals:

```ros2 launch micromate_ros2_driver micromate_with_custom_ros2_bringup.launch.py use_mock_hardware:=true```

```ros2 launch custom_needle_moveit move_group.launch.py```

In another terminal, run Slicer by using the command: ```ros2 run slicer_ros2_module slicer```


The following steps are completed in Slicer, with any commands run in the Slicer Python Console:

Run the trajectory generator: ```exec(open('~/ros2_ws/src/transpedicular_planner.py').read())```

Load in the CT and segmentation files (replace instances of [DATA FOLDER] and [FILE] with the appropriate file paths (e.g. 01_training and verse503)): ```ct, seg = loadCase('~/ros2_ws/src/dataset/[DATA FOLDER]/rawdata/sub-[FILE]/sub-[FILE]_dir-ax_ct.nii.gz', '~/ros2_ws/src/dataset/[DATA FOLDER]/derivatives/sub-[FILE]/sub-[FILE]_dir-ax_seg-vert_msk.nii.gz')```

Open the Segmentation module and click **Show 3D** to visualize the 3D spine model

See vertebrae labels, if desired: ```listSegments(ct, seg)```

Generate trajectories, specifying the segment label: ```p, res = planVertebra(ct, seg, 'Segment_21')```

Clear old trajectories, if necessary: ```clearPlans()```


Open the Slicer ROS module and click **+ Add New Robot** to connect Slicer to the robot description published by the previously launched Micromate driver, then click **Load Robot**

Open the Motion Control module, [EDIT] xy_needle planning group

Perform motion planning and control by running the following:
```exec(open('~/ros2_ws/src/src/slicer_ros2_module/micromate_motion_control.py').read())```

```out = runMicromate(res, 'left', stage='move')```

Plan, Preview, and Execute the robot’s x, y, and tilt (no needle movement) following this command 

```planSide(res, 'left', stage='insert')```

Plan, Preview, and Execute needle insertion following this command 

```planHome(retract_only=True)```

```planHome()```
Home needle and joints

## Acknowledgement
This project is supported by:
* The National Institute of Biomedical Imaging and Bioengineering of the U.S. National Institutes of Health (NIH) under award number R01EB020667, and 3R01EB020667-05S1 (MPI: Tokuda, Krieger, Leonard, and Fuge).  The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH.
* The National Sciences and Engineering Research Council of Canada and the Canadian Institutes of Health Research.

## References
- 3D Slicer Official Page, https://www.slicer.org/
- ROS - Robot Operating System, https://ros.org/
- ROS for Medical Robotics, https://rosmed.github.io/
- Presentations: https://www.youtube.com/watch?v=OZBgR6te3Gs
- Workshops: ISMR 2023 - https://rosmed.github.io/ismr2023/index, ISMR 2024 - https://rosmed.github.io/ismr2024/index
- **Introductory publication:** Connolly L, Deguet A, Leonard S, Tokuda J, Ungi T, Krieger A, Kazanzides P, Mousavi P, Fichtinger G, Taylor RH. Bridging 3D Slicer and ROS2 for Image-Guided Robotic Interventions. Sensors (Basel). 2022 Jul 17;22(14):5336. doi: 10.3390/s22145336. PMID: [35891016](https://pubmed.ncbi.nlm.nih.gov/35891016/); PMCID: [PMC9324680](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9324680/).
- **Most-recent publication:** Connolly L, Kumar AS, Mehta KK, Al-Zogbi L, Kazanzides P, Mousavi P, Fichtinger G, Krieger A, Tokuda J, Taylor RH, Leonard S, Deguet A. SlicerROS2: A Research and Development Module for Image-Guided Robotic Interventions. IEEE Trans Med Robot Bionics. 2024 Nov;6(4):1334-1344. doi: 10.1109/TMRB.2024.3464683. [PMID: To be added]; [PMCID: To be added].

If you use SlicerROS2 and find it helpful for your research, please use the 2024 publication for citation: 

```
@article{connolly2024slicerros2,
  title={SlicerROS2: A Research and Development Module for Image-Guided Robotic Interventions},
  author={Connolly, Laura and Kumar, Aravind S and Mehta, Kapi Ketan and Al-Zogbi, Lidia and Kazanzides, Peter and Mousavi, Parvin and Fichtinger, Gabor and Krieger, Axel and Tokuda, Junichi and Taylor, Russell H and others},
  journal={IEEE Transactions on Medical Robotics and Bionics},
  year={2024},
  publisher={IEEE}
}
```

## Other

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rosmed/slicer_ros2_module)
[![Ubuntu 24.04 Jazzy Slicer 5.10](https://github.com/rosmed/slicer_ros2_module/actions/workflows/ubuntu-24.04-jazzy-slicer-5.10.yml/badge.svg)](https://github.com/rosmed/slicer_ros2_module/actions/workflows/ubuntu-24.04-jazzy-slicer-5.10.yml)
[![Ubuntu 24.04 Jazzy Slicer 5.12](https://github.com/rosmed/slicer_ros2_module/actions/workflows/ubuntu-24.04-jazzy-slicer-5.12.yml/badge.svg)](https://github.com/rosmed/slicer_ros2_module/actions/workflows/ubuntu-24.04-jazzy-slicer-5.12.yml)
