import sys
import io

class LogFilter:
    def __init__(self, stream, ignore_keywords):
        self.stream = stream
        self.ignore_keywords = ignore_keywords
        self.buffer = ""

    def write(self, data):
        # バッファリングして行単位で処理するか、そのまま流すか
        # プログレスバー()への対応が重要
        
        # 単純なフィルタリング: データブロックの中にキーワードが含まれていたら捨てる？
        # しかし data は "Epoch 1/10\n" のように来ることもあれば、1文字ずつのこともある。
        # 安全策: 行単位で判定したいが、\r の更新を阻害したくない。
        
        # アプローチ: 
        # C++のログは行単位で来る確率が高い。
        # プログレスバーは \r を含む。
        
        if any(k in data for k in self.ignore_keywords):
            return
            
        # 念のため、改行で分割してチェック（dataが複数行を含む場合）
        if '\n' in data:
            lines = data.split('\n')
            for i, line in enumerate(lines):
                if any(k in line for k in self.ignore_keywords):
                    continue
                # 最後の要素以外は改行をつける
                if i < len(lines) - 1:
                    self.stream.write(line + '\n')
                else:
                    self.stream.write(line)
        else:
            self.stream.write(data)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

def install_log_filter():
    ignore_keywords = [
        "computation_placer.cc",
        "cuda_dnn.cc",
        "service.cc",
        "E0000",
        "I0000",
        "WARNING",
        "oneDNN",
        "successful NUMA",
        "AgPlacer",
        "tensorflow",
        "xla_compiler"
    ]
    
    # 標準出力と標準エラー出力をフック
    # sys.stdout = LogFilter(sys.stdout, ignore_keywords) # stdoutはプログレスバーが多いので慎重に
    sys.stderr = LogFilter(sys.stderr, ignore_keywords) # TFのログは主にstderrに出る

if __name__ == "__main__":
    install_log_filter()
    print("Normal output")
    sys.stderr.write("I0000 This should be ignored\n")
    sys.stderr.write("Normal stderr\n")
