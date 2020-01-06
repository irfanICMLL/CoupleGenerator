import os
from PIL import Image
orgi_dir='/data/nhli/irfan/zhongqi/lsun/data/'
out_dir='/data/nhli/irfan/zhongqi/lsun/test_pic/'
if not os.path.exists(out_dir): os.mkdir(out_dir)
i=1
for root, dirs, files in os.walk(orgi_dir):
    for file in files:
        if 'webp' in file:
            i=i+1
            im=Image.open(os.path.join(root,file))
            name=root.split('/')[-1]
            x=file.replace('webp','jpg')
            im.imresize(64,64)
            im.save(out_dir+str(i)+'.png')
            print(os.path.join(root,file))