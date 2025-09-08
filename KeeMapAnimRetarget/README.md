# Make sure you are in Pose Mode to see the UI!!!!!!
# Keemap-Blender-Rig-ReTargeting-Addon
Blender Rig Retargeting Addon allows mapping motions of one rig to another.  Works with ANY rig and allows user to map bones from one rig to another and save mapping files out to hard drive.  This script is tested and working on blender 2.83, if a newer version of blender breaks the script I'm sorry.

Installation Procedure:
Download the zip file in the root folder NOT the entire source code tree.  Go to Blender-->edit(pull down from menus at top)--> Preferences, then click on the add ons button on the left of the ui.  Click on the Install button.  Select the zip file you have downloaded.  When the script shows up on the list check the box to enable it.  Make certain you DO NOT download this entire source code tree as a zip and try and install that, if you do, make sure to unzip it until you get to the file 'KeeMap Rig Transfer Addon.zip' and DO NOT UNZIP THIS FILE.  You must select this ZIP file still zipped to install into blender.

# Reasons for failure to install!!!! 
when the addon doesn't show up it's usually one of two reasons:
1. The zip file you are installing is a zip of a zip, open the zip file and make sure there isn't another zip inside the one you are trying to install.
2. After installing you are not in pose mode so you don't see the add on tag.

Tutorial:
Here is a tutorial video with instructions on exactly how to use the script:

https://youtu.be/EG-VCMkVpxg

UI:

![Image of the Blender UI](https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon/blob/main/Images/KeeMapUI.jpg)

In the above image is the UI for the GUI in blender when the script is installed.

## WorkFlow/Quick Start

After getting your source and destination rigs in the the same blend file enable addon, select both character's rigs in object mode and change to pose mode.

Select a bone in the source rig, and select the eyedropper in the source rig name box.
Select a bone in the destination rig and select eyedropper in the destination rig box.

Select the browse in the bone mapping file dialog and put a name of a file you want to save your settings to (ie rig2rig.json or mysetting.json or some other).

Click the Save bone mapping file (do this regularly to save your settings), at any time you can click read in to restore your last save or use this file on another blend etc.

Select New to create a bone.

Click the root bone in the source rig and select Get Name to populate it's name.
select the root bone in destination rig and press Get Name.

Select Test to see what happens.  rotate the destination rig's root bone to be the correct angle and press calc correction.

Create a new bone repeating the above process working your way through the entire rig.  
Use Test All to position the entire rig to match the source.
Scrub the timeline back and forth and test all to make sure the transfer works.

When you are done, put a start from and number of samples into the transfer settings and click 'transfer animation' to transfer and keyframe a large animation all in one go.

## Transfer Settings

![Image of the Blender UI](https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon/blob/main/Images/TransferSettings.jpg)

**Starting Frame**: When the 'Transfer Animation from Source to Destination Character' button is pressed this is the starting frame in the timeline to start applying the rig modifications from the source to the destination.

**Number of Samples**: This will be the number of frames in the timeline to transfer.  This is timeline units so if you put a start from of 10 and a number of samples of 20, then the transfer will start at frame 10 and continue until it gets to 30.

**KeyFrame Number**: this is the number of frames to wait between each keyframe. so in the previous example with a start frame of 10 and a number of samples of 20 and a Keyframe number of 5 you will get a transfer and keyframe at 10,15,20,25 and 30.

**Source Rig Name**: Place the Name of the armature that is already keyframed with the animation you wish to transfer.

**Destination Rig Name**:   Place the name of the armature that you will map all of the tranformations of the source rig on to.

**Bone Mapping File**:  Browse to an existing or put the name of a file in this location using the browse button to save your bone mapping work to or read from.

**Read in Bone Mapping File**: press this button to read in all settings from the file to your current settings.  This will BLOW AWAY all settings in the file and replace them with the current settings you have.  EVERYTHING in the KeeMap gui is saved to the file including all check boxes and text fields for both bones and start frames etc.

**Save Bone Mapping File**:  Press this button to save out all bones and their mapping as well as all other settings you've made in the gui to the selected Bone Mapping File.

**Transfer Animation from Source to Destination Character**:  Press this button to transfer an animation from the start frame until the number of samples has been reached.  For example with a start frame of 10 and a number of samples of 20 and a Keyframe number of 5 you will get a transfer and keyframe at 10,15,20,25 and 30.  This transfer can take some time.  A percent complete is printed to the console so you can watch its progress.  Select toggle system console in blender to see this, otherwise just wait a while.


## Bone Mapping:


![Image of the Blender UI](https://github.com/nkeeline/Keemap-Blender-Rig-ReTargeting-Addon/blob/main/Images/BoneMapping.jpg)

**Bone List**: The bone list is a section that contains each bone map.  When you click 'New' it will create a new bone mapping and it's settings will display below.  You can create as many bone maps as you like.  Each bone map will get run IN ORDER when the Transfer Animation button is pressed.  When Rotation is transferred from one bone to another all of it's children will move.  Therefore this list should start with parents and move to the children so no bones are moved erroneously..

**New**: this will add a new bone to the list.

**Remove**: this will remove the selected bone from the list.

**Up**: this moves the selected bone up one in the list making this bone get operated on prior to all bones below it.

**Down**: this will move a bone down the list causing it to be acted on after all bones above it.

**name**: this is a use setable name for the bone to help you identify it.  It can be blank or anything you like and will not effect how the mapping or transfer is done.

**Source Bone Name**:  this is the name of the bone from the source armature from which to get the information for the transfer of rotation or location.

**Destination Bone Name**:  this is the name of the destination bone to which the tranformation will be applied.

**Get Name**:  Press this button to get the selected bones name and auto populate the source or destination bone names for you.  The Destination Rig Name and the Source Rig Names will be used to auto detect which field to place the bone's name into source or destination.  If the name field is empty and you select a source bone, it will be auto populated too for convenience, but if you ever change the name it will not be updated by this button.  Get Name will error if the source or destination rig name is wrong or missing.

**Select**:  This button will select the bonemapping automatically based on the bone that is selected on the destination rig in blender.  This way when you are tweaking settings you can click on the bone on the rig, press this button and the correct bone mapping will be auto selected for you.

**KeyFrame This Bone**:  Select this check box to keyframe this bone.  Unchecking this box will cause this bones rotation and position to NOT be keyframed when any process is run.  Basically this will disable this bone, but may be useful if you want to position some bone prior to positioning and keyframing another.  Not sure why you would want to do this, but this check box is there in case it is needed.

**Set Rotation of Bone**:  Check this box to set a destination bones global rotation data with the source bones global rotation.

**Apply To Axis**:  select which axis to apply the rotation, you have the option of limit the rotation modification to only a few axis of the bone.  In almost all circumstances leave this at 'XYZ'

**Transpose Axis**:  This should never be used, but allows you to take the XYZ from bone and move where the data goes.  I put this in the script just in case some day it is needed, hopefully never.

**Correction Rotation**:  After the bone is positioned globally from the source bone this is the additional rotation to move the bone.  Say your character limbs are disappearing inside it's body because the source rig was skinnier and the large characters arms need to angle outward more.  rotate the destination characters arms until they are more angled out and note which axis you did it in and then put those values in the correction field to correct the problem.

**Calc correction**:  Use this button at your own risk, but basically if you press the test button to position the bone, you can correct the bones position in the 3d view then press this button to 'auto' populate the correction factor with your corrected values.  This button hasn't been tested fully and may have some use, but I recommend saving your bone mapping file prior to using it and if doesn't do what you want read the file back to put the values back and give yourself an undo.

**Set Position of Bone**: Typically this is only used on the 'root' bone of a destination armature to position the character in the same locaiton as the source character.

**Position Type**:  There are two types of position setting Single Bone Offset just takes the destination bone and positions it to the world space of the source bone with a pose space offset.  This is useful for IK rigs where you want to place the foot bone in world space position, and then move it a little in pose space to account for differences in length of limbs.  Pole Bone takes prompts you for the source be the base of the ik chain then moves out one child to make three points, the base of the base bone, the tip of the base bone and the tip of the child bone.  The ratio'd average of the base of the base bone and the tip of the child is created. So for a leg example the hip and the ankle are averaged by the length of the thigh divided by the length of the knee.  The resulting point is the same rough height as the knee, but typically behind it if the knee is bent.  A line is then drawn in 3d space from the point to the knee and then out a a distnace in front of the knee.  The pole bone is placed there.

**Pole Distance** This is the distance out front or behind a pole bone is placed in Pole Bone Mode:

**Correction Position** This is a calculated offset in pose space between the source rig and the destination rig bone to apply when setting position.

**Position Gain** Rarely used gains up or down the distance of all three axis (should no be used, not sure why I put it in there)

**Position Calc Corrction**  when both armatures are in tpose mode click this button to calculate and populate an automatic correction factor for the position.

**Set Scale of Bone**  A solution is needed for rigify where there is a bone which is scaled to curve or close the finger.  The angle of the base of the finger with respect to the tip is used to scale the rigify finger bone control.  To do this you add the tip bone to the calc give it a scale to correct for the amount of curvature you want and off you go.  Click this check box to apply this correction.

**2nd Source Scale Bone** This is the name for the source rig bone name the angle with respect to the other source bone will be used to scale the destination bone.  The angle between the two source bones are used to scale the dest bone.  Mainly used for rigify where the angle between the tip of the finger and the base finger bone is used to scale the finger control bone so scaling the control bone bends the finger.

**Apply to Axis**  Which axis do you wish to apply the scale to on the control bone.  In rigify the scale is applied to the Y axis of the control bone, so that's the default.

**Scale Gain**Amount of gain to apply to scale affect to adjust for too much or two little curvature.

**Test**:  Press this to position the destination bone according to your settings to see what your settings do.   VERY USEFUL, use this constantly to test everything as you map all the bones in your character.

**Test All**: this is the same as selecting each bone in the bone list in turn and pressing the 'Test' button.

**KeyFrame Test**:  This will keyframe what you you test.  In essence you can move the timeline along and with this checkbox checked, press the test buttons and it will keyframe each test you do.  In this way you can manually run a transfer putting the keyframes manually exactly where you want them.