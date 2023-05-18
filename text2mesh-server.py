# Server and Threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import threading
import uuid
import http
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
prompt = ""
request_file_path = ""

def func_thread_polling_prompt():    
    device_target = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_target)
    print(f"device_target {device_target}")

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

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        karras_steps=[32, 32],
        model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
    )

    while True:
        while not busy_lock.locked():
            if kill_lock.locked():
                return
            time.sleep(0.1)

        global prompt, request_file_path

        print(f"> prompt selected: '{prompt}'")
        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x

        pc = sampler.output_to_point_clouds(samples)[0]

        device = torch.device(device_target)

        print('creating SDF model...')
        name = 'sdf'
        model = model_from_config(MODEL_CONFIGS[name], device)
        model.eval()

        print('loading SDF model...')
        model.load_state_dict(load_checkpoint(name, device))

        import skimage.measure # To avoid AttributeError

        # Produce a mesh (with vertex colors)
        mesh = marching_cubes_mesh(
            pc=pc,
            model=model,
            batch_size=4096,
            grid_size=32, # increase to 128 for resolution used in evals
            progress=True,
        )


        print(f"saving ply on {request_file_path}")
        # Write the mesh to a PLY file to import into some other program.
        with open(request_file_path, 'wb') as f:
            mesh.write_ply(f)

        busy_lock.release()


class Server(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/prompt'):
            self.do_prompt()
        elif self.path.startswith('/files'):
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        else: 
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Not found", "utf-8"))
            return
    
    def do_prompt(self):
        if busy_lock.locked():
            self.send_response(423)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Busy", "utf-8"))
            return
        else:
            global prompt, request_file_path
            
            maybe_prompt = None
            try:
                query = urlparse(self.path).query
                query_obj = parse_qs(query)
                maybe_prompt = query_obj.get('text', None)[0]
            except:
                pass

            print(query, query_obj, maybe_prompt)

            if type(maybe_prompt) != str: 
                self.send_response(400)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(bytes("You have to pass `text` parameter. Example: ?text=green%20tree", "utf-8"))
                return

            # Set a prompt
            prompt = maybe_prompt

            request_uuid = uuid.uuid4()
            request_file_path = f"files/{request_uuid}.ply"

            self.send_response(200)
            self.send_header("Content-type", "text/json")
            self.end_headers()
            self.wfile.write(bytes(f"{{\"id\":\"{request_uuid}\",\"file\":\"{request_file_path}\"}}", "utf-8"))

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