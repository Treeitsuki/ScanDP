# Environment Setup Notes

## Python virtual environment

Install the project dependencies into a `uv`-managed virtual environment:

```bash
uv sync
source .venv/bin/activate
uv pip install --extra-index-url https://rospypi.github.io/simple rospy-all
uv pip install setuptools==81.0.0
```

## ROS workspace

Build the catkin workspace after sourcing ROS:

```bash
source /opt/ros/noetic/setup.bash
catkin build
source devel/setup.bash
```

## Notes

- Always activate the virtual environment *after* sourcing the ROS setup files.
- Reference: [Using `rospy` inside a Python 3 virtualenv](https://qiita.com/otamasan/items/7ac7732a5c3d47ec3028)
