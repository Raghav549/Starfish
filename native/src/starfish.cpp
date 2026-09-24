#include "starfish.hpp"
namespace starfish {
int crossing_number(const uint8_t* p,int w,int h,int x,int y){
 if(x<=0||y<=0||x>=w-1||y>=h-1)return 0;
 const int dx[8]={-1,0,1,1,1,0,-1,-1},dy[8]={-1,-1,-1,0,1,1,1,0};
 int s[8]; for(int i=0;i<8;i++) s[i]=p[(y+dy[i])*w+(x+dx[i])] ? 1:0;
 int t=0; for(int i=0;i<8;i++) if(s[i]!=s[(i+1)%8]) ++t; return t/2;
}
std::vector<int> crossing_numbers(const uint8_t* p,int w,int h){
 std::vector<int> out((size_t)w*h); for(int y=1;y<h-1;y++)for(int x=1;x<w-1;x++)out[(size_t)y*w+x]=crossing_number(p,w,h,x,y); return out;
}}