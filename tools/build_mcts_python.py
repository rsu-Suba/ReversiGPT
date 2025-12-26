import os
import struct
import base64
import zlib
import re

def minify_python(code):
    lines = []
    for line in code.split('\n'):
        line_content = line.rstrip()
        if not line_content.strip(): continue
        if line_content.strip().startswith('#'): continue
        lines.append(line_content)
    return '\n'.join(lines)

def main():
    weights_path = "submission/weights.bin"
    output_path = "submission/submission.py"

    if not os.path.exists(weights_path):
        return

    with open(weights_path, "rb") as f:
        weights_data = f.read()
    
    compressed_data = zlib.compress(weights_data, level=9)
    b85_str = base64.b85encode(compressed_data).decode('ascii')

    part1 = r"""import sys
import os
import time
import math
import zlib
import base64
import struct
import numpy as np

W_B85 = "__WEIGHTS_PLACEHOLDER__"

class NanoModel:
    def __init__(self):
        self.d = 32
        self.h = 4
        self.l = 3
        self.w = self._L()

    def _L(self):
        try:
            r = zlib.decompress(base64.b85decode(W_B85))
            r = np.frombuffer(r, dtype=np.float16).astype(np.float32)
        except: return None

        w = {}
        p = 0
        d = self.d
        
        def g(s):
            nonlocal p
            z = np.prod(s)
            v = r[p:p+z].reshape(s)
            p += z
            return v

        w['i0'] = g((2, d))
        w['i1'] = g((d,))
        w['r'] = g((8, d))
        w['c'] = g((8, d))
        
        w['B'] = []
        for _ in range(self.l):
            b = {}
            b['q0'] = g((d, d)); b['q1'] = g((d,))
            b['k0'] = g((d, d)); b['k1'] = g((d,))
            b['v0'] = g((d, d)); b['v1'] = g((d,))
            b['o0'] = g((d, d)); b['o1'] = g((d,))
            b['n0'] = g((d,));  b['n1'] = g((d,))
            b['f0'] = g((d, 4*d)); b['f1'] = g((4*d,))
            b['f2'] = g((4*d, d)); b['f3'] = g((d,))
            b['m0'] = g((d,));     b['m1'] = g((d,))
            w['B'].append(b)
            
        w['p0'] = g((d, 1)); w['p1'] = g((1,))
        w['v0'] = g((d, 1)); w['v1'] = g((1,))
        return w

    def predict(self, bb):
        w = self.w
        d = self.d
        
        x = np.zeros((64, 2), dtype=np.float32)
        x[:, 0] = np.array([(bb[0] >> i) & 1 for i in range(64)], dtype=np.float32)
        x[:, 1] = np.array([(bb[1] >> i) & 1 for i in range(64)], dtype=np.float32)
        
        x = np.matmul(x, w['i0']) + w['i1']
        
        idx = np.arange(64)
        x += w['r'][idx // 8] + w['c'][idx % 8]
        
        for b in w['B']:
            res = x.copy()
            
            m = np.mean(x, axis=-1, keepdims=True)
            v = np.var(x, axis=-1, keepdims=True)
            xn = b['n0'] * (x - m) / np.sqrt(v + 1e-6) + b['n1']
            
            h = self.h
            dk = d // h
            
            q = np.matmul(xn, b['q0']) + b['q1']
            k = np.matmul(xn, b['k0']) + b['k1']
            v = np.matmul(xn, b['v0']) + b['v1']
            
            q = q.reshape(64, h, dk).transpose(1, 0, 2)
            k = k.reshape(64, h, dk).transpose(1, 0, 2)
            v = v.reshape(64, h, dk).transpose(1, 0, 2)
            
            s = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(dk)
            
            es = np.exp(s - np.max(s, axis=-1, keepdims=True))
            a = es / np.sum(es, axis=-1, keepdims=True)
            
            ao = np.matmul(a, v).transpose(1, 0, 2).reshape(64, d)
            mo = np.matmul(ao, b['o0']) + b['o1']
            
            x = res + mo
            res = x.copy()
            
            m = np.mean(x, axis=-1, keepdims=True)
            v = np.var(x, axis=-1, keepdims=True)
            xn = b['m0'] * (x - m) / np.sqrt(v + 1e-6) + b['m1']
            
            ff1 = np.matmul(xn, b['f0']) + b['f1']
            ff1 = 0.5 * ff1 * (1 + np.tanh(0.79788456 * (ff1 + 0.044715 * ff1**3)))
            ff2 = np.matmul(ff1, b['f2']) + b['f3']
            
            x = res + ff2
            
        p = np.matmul(x, w['p0']) + w['p1']
        p = p.flatten()
        
        vh = np.mean(x, axis=0)
        vl = np.matmul(vh, w['v0']) + w['v1']
        v = np.tanh(vl)[0]
        
        return p, v
"""

    part2 = r"""
class Bitboard:
    def __init__(self):
        self.P = [9,-3,3,3,3,3,-3,9,-3,-5,0,0,0,0,-5,-3,3,0,1,1,1,1,0,3,3,0,1,1,1,1,0,3,3,0,1,1,1,1,0,3,3,0,1,1,1,1,0,3,-3,-5,0,0,0,0,-5,-3,9,-3,3,3,3,3,-3,9]
        self.MASK = 0xFFFFFFFFFFFFFFFF
        self.reset()
        
    def reset(self):
        self.b = 0x0000001008000000
        self.w = 0x0000000810000000
        self.cp = 0 
        
    def get_legal_moves(self, p):
        my = self.b if p == 0 else self.w
        op = self.w if p == 0 else self.b
        blank = ~(my | op) & self.MASK
        
        h = op & 0x7E7E7E7E7E7E7E7E
        v = op & 0x00FFFFFFFFFFFF00
        a = op & 0x007E7E7E7E7E7E00
        
        m = 0
        for d, k in [(1,h),(-1,h),(8,v),(-8,v),(7,a),(-7,a),(9,a),(-9,a)]:
            t = (my << d) & k if d > 0 else (my >> -d) & k
            for _ in range(5):
                t |= (t << d) & k if d > 0 else (t >> -d) & k
            m |= (t << d) & blank if d > 0 else (t >> -d) & blank
        return m
        
    def make_move(self, idx, p):
        my = self.b if p == 0 else self.w
        op = self.w if p == 0 else self.b
        bit = 1 << idx
        flip = 0
        h = op & 0x7E7E7E7E7E7E7E7E
        v = op & 0x00FFFFFFFFFFFF00
        a = op & 0x007E7E7E7E7E7E00

        for d, k in [(1,h),(-1,h),(8,v),(-8,v),(7,a),(-7,a),(9,a),(-9,a)]:
            t = 0
            r = (bit << d) if d > 0 else (bit >> -d)
            while r & k:
                t |= r
                r = (r << d) if d > 0 else (r >> -d)
            if r & my:
                flip |= t

        n_my = my | bit | flip
        n_op = op ^ flip
        if p == 0: self.b = n_my; self.w = n_op
        else: self.w = n_my; self.b = n_op
        self.cp = 1 - self.cp

    def move_pass(self):
        self.cp = 1 - self.cp
        
    def count(self):
        return bin(self.b).count('1'), bin(self.w).count('1')

    def heuristic(self, p):
        s = 0
        for i in range(64):
            if self.b & (1 << i): s += self.P[i] if p == 0 else -self.P[i]
            elif self.w & (1 << i): s += self.P[i] if p == 1 else -self.P[i]
        return s * 0.01

class Node:
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.N = 0; self.W = 0; self.Q = 0; self.P = prior
        self.is_expanded = False
        
    def select(self, c=1.0):
        bc = None
        bs = -float('inf')
        for a, ch in self.children.items():
            u = c * ch.P * math.sqrt(self.N + 1) / (1 + ch.N)
            s = ch.Q + u
            if s > bs: bs = s; bc = (a, ch)
        return bc

    def expand(self, logits, moves):
        exps = [math.exp(logits[m]) for m in moves]
        s = sum(exps)
        probs = [e / s for e in exps]
        for i, m in enumerate(moves):
            if m not in self.children:
                self.children[m] = Node(parent=self, prior=probs[i])
        self.is_expanded = True

    def update(self, v):
        self.N += 1
        self.W += v
        self.Q = self.W / self.N

class MCTS:
    def __init__(self, model):
        self.model = model
        
    def search(self, bb, tl):
        root = Node()
        lm = bb.get_legal_moves(bb.cp)
        moves = [i for i in range(64) if (lm >> i) & 1]
        
        if not moves: return -1
            
        inp = (bb.b, bb.w) if bb.cp == 0 else (bb.w, bb.b)
        pol, val = self.model.predict(inp)
        val = val + bb.heuristic(bb.cp)
        
        root.expand(pol, moves)
        root.update(val)
        
        st = time.time()
        while time.time() - st < tl:
            node = root
            sbb = Bitboard()
            sbb.b = bb.b; sbb.w = bb.w; sbb.cp = bb.cp
            path = []
            
            while node.is_expanded and node.children:
                pair = node.select()
                if pair is None: break
                a, node = pair
                path.append(node)
                sbb.make_move(a, sbb.cp)
            
            mm = sbb.get_legal_moves(sbb.cp)
            om = sbb.get_legal_moves(1 - sbb.cp)
            
            v = 0
            if mm == 0 and om == 0:
                b, w = sbb.count()
                if sbb.cp == 0: v = 1.0 if b > w else (-1.0 if w > b else 0.0)
                else: v = 1.0 if w > b else (-1.0 if b > w else 0.0)
            else:
                if mm == 0:
                    sbb.move_pass()
                    inp = (sbb.b, sbb.w) if sbb.cp == 0 else (sbb.w, sbb.b)
                    _, val = self.model.predict(inp)
                    v = -(val + sbb.heuristic(sbb.cp))
                else:
                    vm = [i for i in range(64) if (mm >> i) & 1]
                    inp = (sbb.b, sbb.w) if sbb.cp == 0 else (sbb.w, sbb.b)
                    pol, val = self.model.predict(inp)
                    v = val + sbb.heuristic(sbb.cp)
                    node.expand(pol, vm)
            
            for node in reversed(path):
                v = -v
                node.update(v)
            
        bn = -1
        bm = -1
        for a, ch in root.children.items():
            if ch.N > bn: bn = ch.N; bm = a
        return bm
"""

    part3 = r"""
def main():
    model = NanoModel()
    mcts = MCTS(model)
    bb = Bitboard()
    try:
        l = sys.stdin.readline()
        if not l: return
        my = int(l)
        sys.stdin.readline()
    except: my = 0
    
    while True:
        bb.reset()
        try:
            b = 0; w = 0
            for r in range(8):
                l = sys.stdin.readline().strip()
                for c, h in enumerate(l):
                    if h == '0': b |= (1 << (r * 8 + c))
                    elif h == '1': w |= (1 << (r * 8 + c))
            bb.b = b; bb.w = w
            
            l = sys.stdin.readline()
            if not l: break
            ac = int(l)
            for _ in range(ac): sys.stdin.readline()
            
            bb.cp = my
            pc = bin(bb.b | bb.w).count('1')
            tl = 1.8 if pc <= 4 else 0.13
            
            best = mcts.search(bb, tl)
            
            if best == -1: print("pass")
            else:
                r = best // 8; c = best % 8
                print(f"{chr(ord('a') + c)}{r + 1}")
            sys.stdout.flush()
        except: break

if __name__ == "__main__":
    main()
"""
    
    code_part1 = minify_python(part1.replace("__WEIGHTS_PLACEHOLDER__", b85_str))
    code_part2 = minify_python(part2)
    code_part3 = minify_python(part3)
    
    final_code = code_part1 + "\n" + code_part2 + "\n" + code_part3
    
    with open(output_path, "w") as f:
        f.write(final_code)
        
    print(f"Generated {output_path}")
    print(f"Total size: {len(final_code)} chars")

if __name__ == "__main__":
    main()
