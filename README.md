# stress-monitor
Monitors your stress level by measuring the diameter of your pupils via webcam (optional GSR measurement via Raspberry Pi)

It works by sampling frames from the webcam. After locating face and eyes on those frames, it estimates the pupil diameter relative to the iris diameter, thus providing a measurement of the pupil dilation as a percent of saturation (i.e., when pupil is as big as the iris saturation is at 100%).

For control and calibration purposes, the system obtains additional measurements from a server installed on a Raspberry Pi equiped with an ADC. Upon request, this server provides galvanic-skin response (GSR) measurements; a well established way of measuring stress. Additionally, the server provides information about the illumination of the room, which permits to discount the luminance factor from the pupil dilation measurement.

![Alt text](/doc/stress-monitor.png?raw=true "GSR and luminance measurements provided by Rasperry Pi data server (optional)"