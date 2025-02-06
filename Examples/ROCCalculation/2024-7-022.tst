BEGIN_FILE
	Version 1.0
	Collection CellScaleSquish

BEGIN_BLOCK

BEGIN_DEVICES
1	MicroTester	COM9	AutoDetectCOM	Temperature	37
3	KeyenceGT		AutoDetectCOM
4	KeyenceGT		AutoDetectCOM	Axis 2	Location Far

BEGIN_CAMERAS
1	DMKCamera	USB	5	25	CameraIndex 0	PixPerUmPerZoom	0.31889	Maxfps	44.8201	Resolution 2048x2048

BEGIN_OPTICS
1	Zoom4XDetent

BEGIN_DISPLAYS
1	Video	1	Live_Video	700	700	302	391	FixedAspectRatio	WithMouseOverlays

BEGIN_CHARTS
1	Displacement_um	FAuto	0	100	Time_S	FAuto	0	3620	Position	600	700
	Series	1	X	(74,151,204)	2

BEGIN_HARDWAREOPTS
TemperatureSetPoint	37
CameraType	DMK33UXCamera
CameraShutter	5
CameraGain	25
CameraResolution	0


BEGIN_CONTROLS
Beam	Circular 58 0.5588 411000
Zoom	4
HoldPhaseControl	0 50 0.75
TemperatureWarnings	0	1
TrackingControl	15 50 5 25
CamerasUsed 2

BEGIN_MULTISET
Name	Axis	ZMode	ZFunction	ZUnits	ZMagnitude	ZPreloadType	ZPreloadMag	StretchDurationSec	RecoveryDurationSec	HoldTimeSec	RestTimeSec	NumReps	DataFreqHz	ImageFreqHz	Tolerance	TolCorOver	
Data1	2	Disp	Ramp	um	200	None	5	20	0	3600	0	1	5	0.1	10	0	
