# Server and Threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import threading
import uuid
import http
import json
from urllib.parse import urlparse, parse_qs

# Setup model
import sys
import torch

from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud


from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as shap_e_diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

from PIL import Image
import matplotlib.pyplot as plt

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

# Server Setup
kill_lock = threading.Lock()
busy_lock = threading.Lock()

# global variables
ply_request = {
    "prompt": "",
    "uuid": "",
    "grid_size": 32,
    "sampler": "fast",
    "file": ""
}

sampler_option = ["fast", "slow", "shap_e"]

def func_thread_polling_prompt():    
    device_target = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_target)
    print(f"device_target {device_target}")

    # shap_e
    print('creating shap-e base model...')
    shape_xm = load_model('transmitter', device=device)
    shape_model = load_model('text300M', device=device)
    shape_diffusion = shap_e_diffusion_from_config(load_config('diffusion'))

    # Model Setup
    print('creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    sampler_fast = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        karras_steps=[32, 32],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )

    sampler_slow = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )

    print('creating SDF model...')
    name = 'sdf'
    model = model_from_config(MODEL_CONFIGS[name], device)
    model.eval()

    print('loading SDF model...')
    model.load_state_dict(load_checkpoint(name, device))
    
    while True:
        while not busy_lock.locked():
            if kill_lock.locked():
                return
            time.sleep(0.1)

        global ply_request

        try:
            if ply_request.get('sampler') == 'shap_e':
                print(f"> (shap_e) prompt selected: '{ply_request.get('prompt')}'")

                batch_size = 1
                guidance_scale = 15.0

                latents = sample_latents(
                    batch_size=batch_size,
                    model=shape_model,
                    diffusion=shape_diffusion,
                    guidance_scale=guidance_scale,
                    model_kwargs=dict(texts=[ply_request.get('prompt')] * batch_size),
                    progress=True,
                    clip_denoised=True,
                    use_fp16=True,
                    use_karras=True,
                    karras_steps=64,
                    sigma_min=1e-3,
                    sigma_max=160,
                    s_churn=0,
                )

                # Example of saving the latents as meshes.
                from shap_e.util.notebooks import decode_latent_mesh

                request_file_path = ply_request.get('file')
                print(f"saving ply on {request_file_path}")

                for i, latent in enumerate(latents):
                    print(f"saving ply with {i}")
                    t = decode_latent_mesh(shape_xm, latent).tri_mesh()
                    with open(request_file_path, 'wb') as f:
                        t.write_ply(f)
            else:
                if ply_request.get('sampler') == 'slow':
                    sampler = sampler_slow
                else:
                    sampler = sampler_fast

                print(f"> (point_e) prompt selected: '{ply_request.get('prompt')}'")
                # Produce a sample from the model.
                samples = None
                for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[ply_request.get('prompt')]))):
                    samples = x

                pc = sampler.output_to_point_clouds(samples)[0]

                import skimage.measure # To avoid AttributeError

                # Produce a mesh (with vertex colors)
                mesh = marching_cubes_mesh(
                    pc=pc,
                    model=model,
                    batch_size=4096,
                    grid_size=ply_request.get('grid_size'), # increase to 128 for resolution used in evals
                    progress=True,
                )


                request_file_path = ply_request.get('file')
                print(f"saving ply on {request_file_path}")
                # Write the mesh to a PLY file to import into some other program.
                with open(request_file_path, 'wb') as f:
                    mesh.write_ply(f)
        except Exception as ex:
            print(ex)
            print(f"Fail: {ply_request}")
        except:
            print(f"Fail: {ply_request}")

        busy_lock.release()


class Server(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/prompt'):
            self.do_prompt()
        elif self.path.startswith('/files'):
            response = http.server.SimpleHTTPRequestHandler.do_GET(self)
        else: 
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Not found", "utf-8"))
            return
    
    def send_bad_request(self, msg):
        self.send_response(400)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(bytes(msg, "utf-8"))

    def do_prompt(self):
        if busy_lock.locked():
            self.send_response(423)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Busy", "utf-8"))
            return
        else:
            global ply_request
            
            maybe_prompt = None
            query = urlparse(self.path).query
            query_obj = parse_qs(query)
            
            maybe_prompt = query_obj.get('text', None)[0] if query_obj.get('text', None) != None else None
            grid_size = int(query_obj.get('grid_size', None)[0]) if query_obj.get('grid_size', None) != None else 32
            sampler = query_obj.get('sampler', None)[0] if query_obj.get('sampler', None) != None else 'fast'

            print(query, query_obj, maybe_prompt, grid_size, sampler)

            if type(maybe_prompt) != str: 
                self.send_bad_request("You have to pass `text` parameter. Example: ?text=green tree")
                return

            if type(grid_size) != int or grid_size < 16 or grid_size > 128: 
                self.send_bad_request("grid_size param must be a integer from 16 to 128. Example: ?text=green tree")
                return

            if type(sampler) != str or not sampler in sampler_option: 
                self.send_bad_request(f"sampler param options available: {sampler_option}")
                return

            # Set a prompt
            ply_request["prompt"] = maybe_prompt
            ply_request["uuid"] = str(uuid.uuid4())
            ply_request["grid_size"] = grid_size
            ply_request["sampler"] = sampler
            ply_request["file"] = f"files/{ply_request.get('uuid')}.ply"

            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(ply_request, indent = 2), "utf-8"))

            busy_lock.acquire()

if __name__ == "__main__":        
        
    hostName = "localhost"
    serverPort = 8080

    thread_polling_prompt = threading.Thread(target=func_thread_polling_prompt)
    thread_polling_prompt.start()

    webServer = HTTPServer((hostName, serverPort), Server)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
    
    kill_lock.acquire()
    thread_polling_prompt.join()