<h2 align=center>digit recognition AI</h2>
comes with the full tools for training and a website to test the model out.

## Training:
install the required python libraries using the following command: 

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 matplotlib
```
then run [training.py](./training.py):

```shell
python training.py
```
## Testing:
install the required python libraries using the following command: 

```shell
pip install flask_cors torch torchvision pillow flask  
```
then run the [backend](./backend.py):

```shell
python backend.py 
```
then open the [index.html](./index.html) file in your browser.