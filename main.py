import streamlit as st  # Streamlit for building the UI
import torch.nn as nn  # PyTorch neural network module namespace
import inspect  # Python inspect to read signatures of classes/functions
import torch  # PyTorch core
import onnx
import onnx2torch

col1, col2, col3 = st.columns([3, 5, 7])  # create two columns in the Streamlit layout with relative widths

st.subheader('layers')  # render a subheader labeled 'layers'

# --- Help UI: sidebar and main-page guidance (added, no logic changes) ---

st.info("Steps: 1) Filter layers → 2) Click layer → 3) Fill dialog → 4) Apply → 5) Export/Test on the right.")

class Residual(nn.Module):  # define a custom Residual module
 def __init__(self):
  super().__init__()  # initialize parent nn.Module
  self.alpha = nn.Parameter(torch.rand(1))  # trainable parameter alpha initialized randomly
  self.beta = nn.Parameter(torch.rand(1))  # trainable parameter beta initialized randomly

 def forward(self, x, y):  # forward pass expects two inputs for the residual connection
  return (x * self.alpha + self.beta)  + y  # apply learned affine transform to x and add y

class Model(nn.Module):  # define a simple container Model
 def __init__(self):
  super().__init__()  # initialize parent
 
 def forward(self, x):  # forward pass for the Model
  y = x  # start with input as initial y

  for attribute in self.children():  # iterate over child modules attached to this model
   if dir(attribute) == dir(Residual()):  # compare attribute interface to a Residual instance's interface
    x = attribute(x, y)  # if it's a Residual-like module, call with (x, y)
    y = x

   else:
    y = attribute(y)  # otherwise call the module with the single tensor y

  return y  # return the final tensor after applying all children
 
model = Model()  # instantiate the Model

# initialize session state entries if they don't exist yet
if 'model' not in st.session_state and 'attributes_applied' not in st.session_state:
 st.session_state['model'] = model  # store the model object in Streamlit session state
 st.session_state['attributes_applied'] = []  # store a list of applied attribute names in session state


@st.dialog("Enter the following parameters")  # declare a Streamlit dialog for entering constructor parameters
def input_popup(keys, attribute, obj, is_imported_layer):
 kwargs = {}  # dictionary to collect constructor keyword arguments from the dialog inputs

 for key in keys:
  argument = st.text_input(key)  # present a text input for each constructor parameter name

  if argument and argument.isdigit():
   kwargs.update({key:int(argument)})  # if the input is digits only, convert to int and store
  
  if argument == 'True' or argument == 'False':
   if argument == 'True':
    argument = True

   if argument == 'False':
    argument = False

   kwargs.update({key:argument})  # if the string is the literal 'True' or 'False', store a bool (note: keeps code as-is)
 
 if st.button('apply layer'):
   number_of_same_attributes = st.session_state['attributes_applied'].count(attribute) + 1 # count how many times this attribute name has been applied
   
   if is_imported_layer:
    setattr(st.session_state['model'], attribute + '_' + str(number_of_same_attributes if number_of_same_attributes > 1 else ''), obj)

   else:
    setattr(st.session_state['model'], attribute + '_' + str(number_of_same_attributes if number_of_same_attributes > 1 else ''), obj(**kwargs))

   st.success('Layer applied successfully')
   # attach the new layer instance to the model object using a generated attribute name and the parsed kwargs

# start the list of available attributes with the custom Residual entry
attributes_and_names = [(Residual, (), 'Residual')]
losses = []

for attribute in dir(nn):  # iterate over attribute names in the torch.nn module
 obj = getattr(nn, attribute) # get the object from nn by name
 
 if 'Loss' not in attribute:  # skip things that include 'Loss' in the name 
  if isinstance(obj, type):  # only consider classes/types
            try:
                sig = inspect.signature(obj)  # attempt to get the constructor signature
                keys = list(sig.parameters.keys())  # collect parameter names from the signature
                 
                attributes_and_names.append((obj, keys, attribute.lower()))  # append the (class, keys, name) tuple
                
            except:
                pass  # ignore any classes that can't be inspected

 else:
   losses.append((attribute))

 
attributes_applied = []  # local list of attributes applied (kept for compatibility with existing UI code)

# New UI: sidebar filter, expander list for layers, and a form-based export section
query = st.sidebar.text_input("Filter layers", value="")  # sidebar text box to filter layer names

with st.sidebar.expander("Help — How to use this app", expanded=True):
    st.markdown("""
    **Quick Start**
    1. Use the "Filter layers" box to quickly find layer types (e.g. `conv`, `batch`, `drop`).
    2. Click any layer button in the "Available Layers" list to open a parameter dialog.
    3. In the dialog, enter constructor parameters (numbers as digits, booleans as `True`/`False`) and click **apply layer** to attach it to the model.
    4. Re-order or remove layers by editing the model object in session state (advanced users).
    5. Use the export form on the right to provide a dummy input shape (comma-separated) and export to ONNX.

    **Examples**
    - Common shape: `1,3,224,224` for a single image batch with 3 channels.
    - To add a 2D convolution: filter `conv`, click `conv2d`, set `in_channels`, `out_channels`, `kernel_size`, then apply.

    **Notes & Tips**
    - If a dialog parameter should be an integer, enter only digits (e.g. `64`).
    - For boolean flags enter `True` or `False` exactly.
    - Use the Filter to reduce scrolling when the layer list is long.
    """)
    st.write("If something goes wrong, check the server logs where Streamlit is running for traceback details.")

    # Additional explanation specifically for the custom Residual layer
    st.markdown("""
    ---
    **About the custom `Residual` layer**

    This app includes a small custom `Residual` module which implements a simple learned residual connection.

    - Signature: `Residual()` — when used, it expects two inputs in the model forward pass: the current layer input `x` and the running state `y`.
    - Computation (in this app): `(x * alpha + beta) + y` where `alpha` and `beta` are trainable parameters initialized randomly.
    - `alpha` scales the incoming `x`, `beta` shifts it, and then the result is added to `y` (the residual connection).

    When to use it:
    - Use `Residual` if you want a learnable skip connection that mixes the incoming activation with the accumulated output.
    - It is different from a plain addition skip because `alpha` and `beta` let the model learn how much of `x` to keep and how to shift it.

    Implementation detail (for advanced users): the app detects Residual-like modules by comparing attribute interfaces to a `Residual()` instance. If a module matches, it will be invoked with `(x, y)` during the model forward pass; otherwise the module is called with a single tensor `y`.
    """)
    
filtered_attributes = [
    (obj, keys, attribute)
    for (obj, keys, attribute) in attributes_and_names
    if query.lower() in attribute.lower()
]  # build a filtered list of attributes whose name contains the query (case-insensitive)

if 'imported_layers' not in st.session_state:
 st.session_state['imported_layers'] = []

if 'loss_used' not in st.session_state:
 st.session_state['loss_used'] = ''

with col1:
    st.subheader("Layer Library")  # header for the left column

    with st.expander("Available Layers", expanded = False):  # an expander that lists available layers
        for i, (obj, keys, attribute) in enumerate(filtered_attributes):  # enumerate filtered layers
           sig = inspect.signature(obj)  # attempt to get the constructor signature
           keys = list(sig.parameters.keys())  # collect parameter names from the signature
                 
           if attribute != 'sequential' and 'module' not in attribute.lower():  # filter out 'sequential' and names containing 'module'
                btn_key = f'layer_btn_{i}_{attribute}'  # generate a unique Streamlit key for the button

                if st.button(attribute, key = btn_key):  # render a button for each layer; when clicked:
                    st.session_state['attributes_applied'].append(attribute)  # record the attribute name in the applied list
                    input_popup(keys, attribute, obj, False)  # open the dialog to supply parameters

    st.subheader('losses')
    
    with st.expander('Available loss functions', expanded = False):
     for loss in losses:
      if st.button(loss):
       st.session_state['loss_used'] = loss

with col2:
    st.subheader("Model Controls")  # header for the right column
    st.write("Use the form below to export or test the model (ONNX).")  # explanatory text

    shape_input = st.text_input("Dummy input shape (comma-separated)", value = "1,3,224,224")  # default shape input
    filename = st.text_input("Enter file name")
    export_btn = st.button(label = 'Export model (.onnx)')

    if export_btn:
            try:
                dims = tuple(int(dim.strip()) for dim in shape_input.split(",") if dim.strip())  # parse the comma-separated dims
                x = torch.rand(dims)  # create a random tensor with the parsed dims
                torch.onnx.export(st.session_state['model'], x, f"{filename}.onnx")  # export model directly to file path
                st.success(f"Exported {filename}.onnx, you can export by clicking download button below")  # show a success message in the UI

                download_btn = st.download_button(label = "Download Model (.onnx)", file_name = f"{filename}.onnx", data = open(f"{filename}.onnx", 'rb').read(), mime = 'application/onnx-model')
                download_btn = st.download_button(label = "Download Model (.data) weights", file_name = f"{filename}.onnx.data", data = open(f"{filename}.onnx", 'rb').read(), mime = 'application/onnx-model')

                if download_btn:
                 st.success('Downloaded successfully, enjoy your first model')

            except Exception as e:
                st.error(f"Export failed: {e}")  # show an error message if export fails

    st.write('use the given below form to import layer')

    uploaded_onnx = st.file_uploader("import layer (.onnx)", type = ["onnx"])
    uploaded_data = st.file_uploader("import data (.data)", type = ["data"], max_upload_size = 10 ** 5)

    if st.button('Import'):
     with open(uploaded_data.name, 'wb') as f:
      f.write(uploaded_data.getbuffer())

     f.close()

     layer = onnx.load(uploaded_onnx)
     layer = onnx2torch.convert(layer)
     st.session_state['imported_layers'].append((uploaded_onnx.name.split('.')[0], layer))

    st.write('use the form given below to train the model')
    
    onnx_name = st.text_input('write model name')
    dataset_name = st.text_input('write dataset name')
    num_epochs = st.text_input('Enter number of epochs')
    learning_rate = st.text_input('Enter learning rate')
    num_batches = st.text_input('Enter number of batches')
    
    if st.button('Generate Training script'):
     
     print(st.session_state['loss_used'])
     code = f"""

import torch.nn as nn
import torch.optim as optim
import pandas as pd
from onnx2pytorch import ConvertModel
import onnx
import torch

model = onnx.load({onnx_name})
model = ConvertModel(model)

optimizer = optim.Adam(lr = {learning_rate}, params = model.parameters())
loss_calculator = nn.{st.session_state['loss_used']}()

ds = pd.read_csv('{dataset_name}')

cols = ds.columns[:-1]

x_dataset = torch.tensor([ds[col] for col in cols], dtype = torch.float32).T
y_dataset = torch.tensor(ds[ds.columns[-1]], dtype = torch.float32)

batch_no = 1

for i in range({num_epochs}):
 print('epoch', i + 1, 'batch no', batch_no, end = '\r')

 for b in range(0, len(ds), {num_batches}):
  batched_x_dataset = x_dataset[b: b + {num_batches}]
  batched_y_dataset = y_dataset[b: b + {num_batches}].unsqueeze(1)

  y_pred = model(batched_x_dataset)
  loss = loss_calculator(y_pred, batched_y_dataset)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  batch_no += 1

 batch_no = 1

torch.save(model.state_dict(), 'logistic_regression.pth')
 """  
     st.success('Generated Training Script successfully, you can download it by clicking Download button below')

     st.download_button(label = "Download Training Script", file_name = f"{onnx_name}_trainer.py", data = code, mime = 'text/x-python')
     

with col3:
     st.subheader('Imported Layers')
     for attribute, layer in st.session_state['imported_layers']:
      sig = inspect.signature(layer)  # attempt to get the constructor signature
      keys = list(sig.parameters.keys())  # collect parameter names from the signature
                 
      btn_key = f'layer_btn_{i}_{attribute}'  # generate a unique Streamlit key for the button

      if st.button(attribute, key = btn_key):  # render a button for each layer; when clicked:
                    st.session_state['attributes_applied'].append(attribute)  # record the attribute name in the applied list
                    input_popup(keys, attribute, layer, True)
 





       


             
