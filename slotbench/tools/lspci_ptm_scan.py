# Scan a tree of saved `lspci -vvnn` dumps (e.g. github.com/linuxhw/LsPCI) and report, per NVIDIA GPU model,
# how many entries show extended config space and how many advertise PCIe Precision Time Measurement.
# Usage: python3 lspci_ptm_scan.py <dir>
import os,re,sys,collections
res=collections.defaultdict(lambda:[0,0,0,[]])  # name -> [n, ext_visible, ptm, example]
for root,_,fs in os.walk(sys.argv[1]):
  for f in fs:
    p=os.path.join(root,f)
    try: t=open(p,errors='ignore').read()
    except: continue
    for blk in re.split(r'\n(?=[0-9a-f]{2,4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d |[0-9a-f]{2}:[0-9a-f]{2}\.\d )',t):
      head=blk.split('\n',1)[0]
      if 'NVIDIA' not in head or not re.search(r'VGA|3D controller',head): continue
      name=re.sub(r'^\S+\s+','',head); name=re.sub(r'\(rev.*','',name).strip()
      r=res[name]; r[0]+=1
      if 'Advanced Error Reporting' in blk or 'Capabilities: [100' in blk: r[1]+=1
      if 'Precision Time Measurement' in blk:
        r[2]+=1; r[3].append(p)
for k,v in sorted(res.items(),key=lambda x:-x[1][0]):
  print(f"{v[0]:5d} ext={v[1]:5d} ptm={v[2]:3d}  {k[:110]}", v[3][:2] if v[2] else '')
