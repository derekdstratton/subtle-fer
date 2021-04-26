# Instructions for Installing the Labeler

(todo: i'll add screenshots for all these)

1. Install Virtualbox from [https://www.virtualbox.org/](https://www.virtualbox.org/). 

2. Save the file `lite test.ova` file to your computer.

3. Open Virtualbox. In the top left, press `file` and select `Import Alliance`.

4. In the pop-up, select the folder icon and locate the `.ova` file from step 2. 
   Then press next.
   
5. On the next screen, press import. It will take some time to import.

6. On the main VirtualBox screen, a new machine called `lite test` should
appear on the left pane. Double click here to run the virtual machine.
   
7. A login screen will appear. Do not change the username and enter the 
   password `osboxes.org`. Proceed to login.

8. The desktop will load with a folder in the middle called `labeling`. 
Open that folder.

9. In that folder there is a file called `run_me`. Double click that file
and the labeler will load and you'll be ready to label!
   
# Instructions for Labeling

todo
   
# Instructions for Submitting Labels

1. Once finished labeling, there are 2 things we'll need to collect. The 
   labels file called `labels.csv` and the images folder called `image masks`.
   Make sure those are both available. The `labels.csv` file is located in
   the `pilot_segments` folder in `labeling`, while `image masks` should be
   visible in `labeling`.
   
2. Right click on the `image masks` folder and select the option `Compress`.

3. Press create, and a file called `image masks.tar.gz` will be created.

4. Please send the files `labels.csv` and `image masks.tar.gz` by
   (how do we want to collect the files. email? not sure how big attachments
   will be. cloud storage?)