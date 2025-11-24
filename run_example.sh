#############################################################
########## Example 1: mouse (+ playroom + lego + hotdog)
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --begin_id 0
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --meshproxy_pitch 0.1 
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --meshproxy_pitch 0.1

#############################################################
########## Example 2: mouse (+ playroom [64 probes] + lego + hotdog)
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --begin_id 0 --numProbes 64
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --meshproxy_pitch 0.1  --numProbes 64
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --meshproxy_pitch 0.1 --numProbes 64

#############################################################
########## Example 3: bunny (+ playroom + lego + hotdog)
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/playroom_lego_hotdog_bunny.ply --probes_path ./models/probes/playroom_lego_hotdog_bunny --mesh ./models/mesh/bunny.ply --begin_id 0
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/playroom_lego_hotdog_bunny.ply --probes_path ./models/probes/playroom_lego_hotdog_bunny --mesh ./models/mesh/bunny.ply --meshproxy_pitch 0.1
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/playroom_lego_hotdog_bunny.ply --probes_path ./models/probes/playroom_lego_hotdog_bunny --mesh ./models/mesh/bunny.ply --meshproxy_pitch 0.1

#############################################################
########## Example 4: dog (+ playroom + lego + hotdog)
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/playroom_lego_hotdog.ply --probes_path ./models/probes/playroom_lego_hotdog_dog --mesh ./models/mesh/dog.ply --begin_id 0
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/playroom_lego_hotdog.ply --probes_path ./models/probes/playroom_lego_hotdog_dog --mesh ./models/mesh/dog.ply --meshproxy_pitch 0.01
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/playroom_lego_hotdog.ply --probes_path ./models/probes/playroom_lego_hotdog_dog --mesh ./models/mesh/dog.ply --meshproxy_pitch 0.01

#############################################################
########## Example 5: ball (+ playroom + lego + hotdog)
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/playroom_lego_hotdog.ply --probes_path ./models/probes/playroom_lego_hotdog_ball --mesh ./models/mesh/ball.ply --begin_id 0
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/playroom_lego_hotdog.ply --probes_path ./models/probes/playroom_lego_hotdog_ball --mesh ./models/mesh/ball.ply --meshproxy_pitch 0.1
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/playroom_lego_hotdog.ply --probes_path ./models/probes/playroom_lego_hotdog_ball --mesh ./models/mesh/ball.ply --meshproxy_pitch 0.1

#############################################################
########## Example 6: penguin (+ Matterport3D_h1zeeAwLh9Z_3)
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/Matterport3D_h1zeeAwLh9Z_3.ply --probes_path ./models/probes/Matterport3D_h1zeeAwLh9Z_3_penguin --mesh ./models/mesh/penguin.ply --begin_id 0 --meshproxy_pitch 0.5
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/Matterport3D_h1zeeAwLh9Z_3.ply --probes_path ./models/probes/Matterport3D_h1zeeAwLh9Z_3_penguin --mesh ./models/mesh/penguin.ply --meshproxy_pitch 0.5
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/Matterport3D_h1zeeAwLh9Z_3.ply --probes_path ./models/probes/Matterport3D_h1zeeAwLh9Z_3_penguin --mesh ./models/mesh/penguin.ply --meshproxy_pitch 0.5

#############################################################
########## Example 7: ball (+ Matterport3D_h1zeeAwLh9Z_3)
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/Matterport3D_h1zeeAwLh9Z_3.ply --probes_path ./models/probes/Matterport3D_h1zeeAwLh9Z_3_ball --mesh ./models/mesh/ball.ply --begin_id 0 
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/Matterport3D_h1zeeAwLh9Z_3.ply --probes_path ./models/probes/Matterport3D_h1zeeAwLh9Z_3_ball --mesh ./models/mesh/ball.ply --meshproxy_pitch 0.1
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/Matterport3D_h1zeeAwLh9Z_3.ply --probes_path ./models/probes/Matterport3D_h1zeeAwLh9Z_3_ball --mesh ./models/mesh/ball.ply --meshproxy_pitch 0.1

#############################################################
########## Example 8: ball (+ drjohnson)
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/drjohnson.ply --probes_path ./models/probes/drjohnson_ball --mesh ./models/mesh/ball.ply --begin_id 0 
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/drjohnson.ply --probes_path ./models/probes/drjohnson_ball --mesh ./models/mesh/ball.ply --meshproxy_pitch 0.1
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/drjohnson.ply --probes_path ./models/probes/drjohnson_ball --mesh ./models/mesh/ball.ply --meshproxy_pitch 0.1

#############################################################
########## Newly Added: 2DGS
########## Example 9: ball (+ garden (2DGS))
#############################################################
python probes_bake.py --W 800 --H 800 --gs_path ./models/2dgs/garden.ply --probes_path ./models/probes/garden_ball --mesh ./models/mesh/ball.ply --begin_id 0 --numProbes 64
python renderer.py --W 960 --H 540 --gs_path ./models/2dgs/garden.ply --probes_path ./models/probes/garden_ball --mesh ./models/mesh/ball.ply --meshproxy_pitch 0.1 --numProbes 64
# or
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/2dgs/garden.ply --probes_path ./models/probes/garden_ball --mesh ./models/mesh/ball.ply --meshproxy_pitch 0.1 --numProbes 64