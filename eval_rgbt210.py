from rgbt import RGBT210

rgbt210 = RGBT210()

# Register your tracker
rgbt210(
    tracker_name="QSTNet",        # Result in paper: 59.9, 37.9
    result_path="./RGBT_workspace/results/RGBT210/rgbt14", 
    bbox_type="ltwh",
    prefix="")

# Evaluate multiple trackers

sr_dict = rgbt210.SR()
print(sr_dict["QSTNet"][0])

pr_dict = rgbt210.PR()
print(pr_dict["QSTNet"][0])

rgbt210.draw_plot(metric_fun=rgbt210.PR)
