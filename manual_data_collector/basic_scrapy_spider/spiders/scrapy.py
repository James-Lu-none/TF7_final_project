import scrapy
# from basic_scrapy_spider.items import QuoteItem
import re
import numpy as np
import pickle
import os
from pathlib import Path
root_path = os.getcwd()
print(root_path)

class playok(scrapy.Spider):
    name = 'playok'
    def __init__(self, name=None, **kwargs):
        self.valid_move=['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'i1', 'j1', 'k1', 'l1', 'm1', 'n1', 'o1', 'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', 'i2', 'j2', 'k2', 'l2', 'm2', 'n2', 'o2', 'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4', 'i4', 'j4', 'k4', 'l4', 'm4', 'n4', 'o4', 'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5', 'i5', 'j5', 'k5', 'l5', 'm5', 'n5', 'o5', 'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6', 'i6', 'j6', 'k6', 'l6', 'm6', 'n6', 'o6', 'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7', 'i7', 'j7', 'k7', 'l7', 'm7', 'n7', 'o7', 'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', 'i8', 'j8', 'k8', 'l8', 'm8', 'n8', 'o8', 'a9', 'b9', 'c9', 'd9', 'e9', 'f9', 'g9', 'h9', 'i9', 'j9', 'k9', 'l9', 'm9', 'n9', 'o9', 'a10', 'b10', 'c10', 'd10', 'e10', 'f10', 'g10', 'h10', 'i10', 'j10', 'k10', 'l10', 'm10', 'n10', 'o10', 'a11', 'b11', 'c11', 'd11', 'e11', 'f11', 'g11', 'h11', 'i11', 'j11', 'k11', 'l11', 'm11', 'n11', 'o11', 'a12', 'b12', 'c12', 'd12', 'e12', 'f12', 'g12', 'h12', 'i12', 'j12', 'k12', 'l12', 'm12', 'n12', 'o12', 'a13', 'b13', 'c13', 'd13', 'e13', 'f13', 'g13', 'h13', 'i13', 'j13', 'k13', 'l13', 'm13', 'n13', 'o13', 'a14', 'b14', 'c14', 'd14', 'e14', 'f14', 'g14', 'h14', 'i14', 'j14', 'k14', 'l14', 'm14', 'n14', 'o14', 'a15', 'b15', 'c15', 'd15', 'e15', 'f15', 'g15', 'h15', 'i15', 'j15', 'k15', 'l15', 'm15', 'n15', 'o15']
        self.games=[]
        self.startpoint = int(153048000)
        self.endpoint =   int(153050000)
        self.segment = int(500)
        self.point=self.startpoint
        self.invalid_cnt=0
        
    def start_requests(self):
        url = f'https://www.playok.com/p/?g=gm{self.point}'
        yield scrapy.Request(url=url,callback=self.start_page)

    def start_page(self, response):
        
        script_text = response.xpath('//script[contains(text(), "k2pback")]//text()').get()
        print("script_text", script_text)
        m_value = re.search(r'k2pback\["m"\] = "(.*?)";', script_text)
        print("here is the data",m_value)
        
        if m_value:
            game=np.full((15,15),-1)
            m_value = m_value.group(1)
            cleaned_moves = re.sub(r'\d+\.\s*', '', m_value)
            separated_moves = cleaned_moves.split()
            print(separated_moves)
            isValid=True
            for i in range(len(separated_moves)-1):
                if(separated_moves[i] not in self.valid_move):
                    print(f"The game {self.point} is invalid")
                    self.invalid_cnt=self.invalid_cnt+1
                    self.point=self.point+1
                    isValid=False
                    break
                game[int(ord(separated_moves[i][0])-96)-1][int(separated_moves[i][1:])-1]=i
        if(isValid):
            self.games.append(game)
            self.point=self.point+1
        if(self.point%self.segment==0):
            
            print(self.games)
            fileName=f"{self.startpoint}_to_{self.point-1}_INV_{self.invalid_cnt}_manual_data"
            dir=os.path.join(root_path,'data')
            with open(os.path.join(dir,fileName), 'wb') as f:
                pickle.dump(self.games, f)
            self.invalid_cnt=0
            self.startpoint=self.point
            self.games.clear()
        if(self.point==self.endpoint):
            print("invalid games:",self.invalid_cnt)
            return
        
        
        yield scrapy.Request(url=f'https://www.playok.com/p/?g=gm{self.point}', callback=self.start_page)
        