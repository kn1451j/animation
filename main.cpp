// ============================================================================
// 2.5D Interactive Animation — main.cpp
// ============================================================================
// Build (macOS):
//   clang -c glad/src/glad.c -Iglad/include -o glad/glad.o   (once)
//   clang++ -std=c++17 main.cpp glad/glad.o -o anim -Iglad/include \
//     -lglfw -lavcodec -lavformat -lavutil -lswscale -lswresample \
//     -framework OpenGL
// Build (Windows): use CMakeLists.txt with vcpkg + pre-built FFmpeg

// ============================================================================
// INCLUDES
// ============================================================================
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// CONFIG
// ============================================================================
namespace cfg {
    constexpr int    WIN_W        = 1920;
    constexpr int    WIN_H        = 1080;
    constexpr double SPRITE_FPS   = 24.0;
    constexpr int    MAX_CYCLES   = 5;
    constexpr int    AUDIO_RATE   = 44100;
    constexpr int    AUDIO_CH     = 2;
    constexpr float  FONT_SIZE_PX = 72.f;
    constexpr int    FONT_ATLAS   = 512;
}

// ============================================================================
// HOMOGRAPHY TRANSFORM
// ============================================================================
// H maps Frame-A pixels → Frame-C pixels (one full L+R cycle).
// For cycle k the sprite transform is H^k in NDC space, embedded in a 4×4
// matrix so OpenGL's perspective division handles the projective component.
//
// H_ndc = M_toNDC · H_pixel · M_toPix
// M_toPix  : NDC → pixel  = [[W/2, 0, W/2], [0, -H/2, H/2], [0,0,1]]
// M_toNDC  : pixel → NDC  = [[2/W, 0, -1],  [0, -2/H,  1],  [0,0,1]]
//
// 4×4 embedding of 3×3 NDC homography [[a,b,c],[d,e,f],[g,h,i]]:
//   col0=(a,d,0,g)  col1=(b,e,0,h)  col2=(0,0,1,0)  col3=(c,f,0,i)
// The w row (g,h,i) causes perspective division handled automatically by GL.

namespace hom {
    static constexpr float H[3][3] = {
        { 0.926354f,  0.000925f, 311.808373f },
        { 0.017365f,  1.066689f, -38.550526f },
        { 0.00003f,   0.0f,        1.0f      }
    };
}

static void mat3Mul(const float A[3][3], const float B[3][3], float C[3][3]) {
    for (int i=0; i<3; ++i)
        for (int j=0; j<3; ++j) {
            C[i][j] = 0.f;
            for (int k=0; k<3; ++k) C[i][j] += A[i][k]*B[k][j];
        }
}

static void homPixToNDC(const float Hp[3][3], float Hn[3][3]) {
    const float W=float(cfg::WIN_W), H=float(cfg::WIN_H);
    float Mp[3][3] = { { W/2,    0,  W/2 },
                       {   0, -H/2,  H/2 },
                       {   0,    0,  1   } };
    float Mn[3][3] = { { 2/W,    0,   -1 },
                       {   0, -2/H,    1 },
                       {   0,    0,    1 } };
    float tmp[3][3];
    mat3Mul(Hp, Mp, tmp);
    mat3Mul(Mn, tmp, Hn);
}

static glm::mat4 embedHom(const float H[3][3]) {
    return glm::mat4(
        glm::vec4(H[0][0], H[1][0], 0.f, H[2][0]),  // col 0
        glm::vec4(H[0][1], H[1][1], 0.f, H[2][1]),  // col 1
        glm::vec4(0.f,     0.f,     1.f, 0.f     ),  // col 2 (z passthrough)
        glm::vec4(H[0][2], H[1][2], 0.f, H[2][2])   // col 3 (translation + w)
    );
}

// Returns the GL model matrix for cycle pair k (0 = identity).
static glm::mat4 cycleTransform(int pair) {
    if (pair == 0) return glm::mat4(1.f);
    float Hn[3][3];
    homPixToNDC(hom::H, Hn);
    float result[3][3] = { {1,0,0},{0,1,0},{0,0,1} };
    for (int i=0; i<pair; ++i) {
        float tmp[3][3];
        mat3Mul(result, Hn, tmp);
        memcpy(result, tmp, sizeof(tmp));
    }
    return embedHom(result);
}

// ============================================================================
// APP STATE
// ============================================================================
enum class AppState { TITLE, INTRO, BACK_POSE, ENTERING_RIGHT, WALKING_RIGHT };
// TODO: WALKING_LEFT, OVERLAY_1/2/3
enum class WalkSub  { IDLE, LEFT_FOOT, RIGHT_FOOT, SHAKE_HEAD };

// ============================================================================
// OPENGL HELPERS
// ============================================================================

// Textured quad — used for sprites, background, video
static const char* VS_SRC = R"glsl(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
uniform mat4 uModel;
void main(){ gl_Position=uModel*vec4(aPos,0.0,1.0); vUV=aUV; })glsl";

static const char* FS_SRC = R"glsl(
#version 330 core
in vec2 vUV; out vec4 frag;
uniform sampler2D uTex; uniform float uAlpha;
void main(){ frag=texture(uTex,vUV); frag.a*=uAlpha; })glsl";

// Font atlas — red channel drives alpha, renders white glyphs
static const char* FS_TEXT = R"glsl(
#version 330 core
in vec2 vUV; out vec4 frag;
uniform sampler2D uTex; uniform float uAlpha;
void main(){
    float a = texture(uTex, vUV).r;
    frag = vec4(1.0, 1.0, 1.0, a * uAlpha);
})glsl";

// Solid color — used for debug placeholders
static const char* VS_PLAIN = R"glsl(
#version 330 core
layout(location=0) in vec2 aPos;
uniform mat4 uModel;
void main(){ gl_Position=uModel*vec4(aPos,0.0,1.0); })glsl";

static const char* FS_PLAIN = R"glsl(
#version 330 core
out vec4 frag; uniform vec4 uColor;
void main(){ frag=uColor; })glsl";

static GLuint compileShader(GLenum t, const char* src) {
    GLuint id = glCreateShader(t);
    glShaderSource(id,1,&src,nullptr); glCompileShader(id);
    GLint ok; glGetShaderiv(id,GL_COMPILE_STATUS,&ok);
    if (!ok){ char l[512]; glGetShaderInfoLog(id,512,nullptr,l); std::cerr<<l<<"\n"; }
    return id;
}

static GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v=compileShader(GL_VERTEX_SHADER,vs);
    GLuint f=compileShader(GL_FRAGMENT_SHADER,fs);
    GLuint p=glCreateProgram();
    glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

struct Quad {
    GLuint vao=0, vbo=0, ebo=0;
    void init() {
        static const float    V[] = {-1,-1,0,1, 1,-1,1,1, 1,1,1,0, -1,1,0,0};
        static const uint32_t I[] = {0,1,2,2,3,0};
        glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo); glGenBuffers(1,&ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,sizeof(V),V,GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(I),I,GL_STATIC_DRAW);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
    }
    void draw() { glBindVertexArray(vao); glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_INT,0); glBindVertexArray(0); }
};

static void drawTex(GLuint prog, Quad& q, GLuint tex,
                    const glm::mat4& M=glm::mat4(1.f), float alpha=1.f) {
    if (!tex) return;
    glUseProgram(prog);
    glUniformMatrix4fv(glGetUniformLocation(prog,"uModel"),1,GL_FALSE,glm::value_ptr(M));
    glUniform1f(glGetUniformLocation(prog,"uAlpha"),alpha);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex);
    glUniform1i(glGetUniformLocation(prog,"uTex"),0);
    q.draw();
}

static GLuint makeTex(int w, int h) {
    GLuint t; glGenTextures(1,&t); glBindTexture(GL_TEXTURE_2D,t);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,nullptr);
    return t;
}

// ============================================================================
// TITLE SCREEN
// ============================================================================
struct TitleScreen {
    stbtt_bakedchar cdata[96];  // ASCII 32..127
    GLuint fontTex  = 0;
    GLuint progText = 0;
    GLuint tvao=0, tvbo=0, tebo=0;

    enum class Phase { TYPING, ELLIPSIS } phase = Phase::TYPING;
    int    charsShown   = 0;
    int    dotCount     = 0;
    double lastCharTime = 0.0;
    double lastDotTime  = 0.0;
    double cursorTimer  = 0.0;
    bool   cursorOn     = true;

    static constexpr double CHAR_DELAY  = 0.11;
    static constexpr double DOT_DELAY   = 0.45;
    static constexpr double CURSOR_RATE = 0.53;
    static constexpr const char* WORD   = "welcome";
    static constexpr int         WORD_LEN = 7;

    bool load(GLuint prog) {
        progText = prog;
        std::vector<std::string> candidates = {
            "fonts/mono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/Library/Fonts/Courier New.ttf",
            "/System/Library/Fonts/Courier New.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/lucon.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        };
        std::vector<uint8_t> fontBuf;
        for (auto& p : candidates) {
            std::ifstream f(p, std::ios::binary);
            if (f) {
                fontBuf.assign(std::istreambuf_iterator<char>(f), {});
                std::cerr << "[title] Font: " << p << "\n";
                break;
            }
        }
        if (fontBuf.empty()) {
            std::cerr << "[title] No font found — title screen disabled\n";
            return false;
        }
        std::vector<uint8_t> bitmap(cfg::FONT_ATLAS * cfg::FONT_ATLAS);
        stbtt_BakeFontBitmap(fontBuf.data(), 0, cfg::FONT_SIZE_PX,
                             bitmap.data(), cfg::FONT_ATLAS, cfg::FONT_ATLAS,
                             32, 96, cdata);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glGenTextures(1,&fontTex); glBindTexture(GL_TEXTURE_2D,fontTex);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_SWIZZLE_R,GL_RED);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_SWIZZLE_G,GL_RED);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_SWIZZLE_B,GL_RED);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_SWIZZLE_A,GL_RED);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RED,cfg::FONT_ATLAS,cfg::FONT_ATLAS,
                     0,GL_RED,GL_UNSIGNED_BYTE,bitmap.data());
        glGenVertexArrays(1,&tvao); glGenBuffers(1,&tvbo); glGenBuffers(1,&tebo);
        glBindVertexArray(tvao);
        glBindBuffer(GL_ARRAY_BUFFER,tvbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,tebo);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
        return true;
    }

    void update(double now) {
        if (now - cursorTimer >= CURSOR_RATE) { cursorOn=!cursorOn; cursorTimer=now; }
        if (phase == Phase::TYPING) {
            if (charsShown < WORD_LEN && now-lastCharTime >= CHAR_DELAY) {
                ++charsShown; lastCharTime=now;
                if (charsShown==WORD_LEN) { phase=Phase::ELLIPSIS; dotCount=1; lastDotTime=now; }
            }
        } else {
            if (now-lastDotTime >= DOT_DELAY) { dotCount=(dotCount%3)+1; lastDotTime=now; }
        }
    }

    float measureWidth(const std::string& s) {
        float x=0,y=0;
        for (char rc : s) {
            unsigned char uc=(unsigned char)rc;
            if (uc<32||uc>=128) continue;
            stbtt_aligned_quad q;
            stbtt_GetBakedQuad(cdata,cfg::FONT_ATLAS,cfg::FONT_ATLAS,uc-32,&x,&y,&q,1);
        }
        return x;
    }

    void render() {
        if (!fontTex) return;
        std::string text(WORD, WORD+charsShown);
        if (phase==Phase::ELLIPSIS) text+=std::string(dotCount,'.');
        else if (cursorOn)          text+='_';
        if (text.empty()) return;

        const float W=float(cfg::WIN_W), H=float(cfg::WIN_H);
        auto ndcX=[&](float px){ return (px/W)*2.f-1.f; };
        auto ndcY=[&](float py){ return 1.f-(py/H)*2.f; };

        float tw=measureWidth(text);
        float sx=(W-tw)*0.5f, sy=H*0.5f+cfg::FONT_SIZE_PX*0.35f;

        struct Vert { float x,y,u,v; };
        std::vector<Vert>     verts;
        std::vector<uint32_t> inds;
        uint32_t vi=0;
        float cx=sx, cy=sy;
        for (char c : text) {
            unsigned char uc=(unsigned char)c;
            if (uc<32||uc>=128) continue;
            stbtt_aligned_quad bq;
            stbtt_GetBakedQuad(cdata,cfg::FONT_ATLAS,cfg::FONT_ATLAS,uc-32,&cx,&cy,&bq,1);
            verts.push_back({ndcX(bq.x0),ndcY(bq.y0),bq.s0,bq.t0});
            verts.push_back({ndcX(bq.x1),ndcY(bq.y0),bq.s1,bq.t0});
            verts.push_back({ndcX(bq.x1),ndcY(bq.y1),bq.s1,bq.t1});
            verts.push_back({ndcX(bq.x0),ndcY(bq.y1),bq.s0,bq.t1});
            inds.insert(inds.end(),{vi,vi+1,vi+2,vi+2,vi+3,vi});
            vi+=4;
        }
        glBindVertexArray(tvao);
        glBindBuffer(GL_ARRAY_BUFFER,tvbo);
        glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(verts.size()*sizeof(Vert)),verts.data(),GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,tebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)(inds.size()*sizeof(uint32_t)),inds.data(),GL_DYNAMIC_DRAW);
        glUseProgram(progText);
        glm::mat4 I(1.f);
        glUniformMatrix4fv(glGetUniformLocation(progText,"uModel"),1,GL_FALSE,glm::value_ptr(I));
        glUniform1f(glGetUniformLocation(progText,"uAlpha"),1.f);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,fontTex);
        glUniform1i(glGetUniformLocation(progText,"uTex"),0);
        glDrawElements(GL_TRIANGLES,(GLsizei)inds.size(),GL_UNSIGNED_INT,0);
        glBindVertexArray(0);
    }
};

// ============================================================================
// VIDEO DECODER  (intro video)
// ============================================================================
struct VideoDecoder {
    AVFormatContext* fmt  = nullptr;
    AVCodecContext*  vctx = nullptr;
    AVCodecContext*  actx = nullptr;
    SwsContext*      sws  = nullptr;
    SwrContext*      swr  = nullptr;
    AVPacket*        pkt  = nullptr;
    AVFrame*         frm  = nullptr;
    int vsi=-1, asi=-1, w=0, h=0;
    bool eof = false;

    std::vector<float>  audioBuf;
    std::atomic<size_t> audioPos{0};

    void preDecodeAudio() {
        if (!actx||!swr) return;
        AVPacket* p=av_packet_alloc(); AVFrame* f=av_frame_alloc();
        av_seek_frame(fmt,-1,0,AVSEEK_FLAG_BACKWARD);
        while (av_read_frame(fmt,p)>=0) {
            if (p->stream_index==asi) {
                avcodec_send_packet(actx,p);
                while (avcodec_receive_frame(actx,f)==0) {
                    int mx=(int)av_rescale_rnd(
                        swr_get_delay(swr,actx->sample_rate)+f->nb_samples,
                        cfg::AUDIO_RATE,actx->sample_rate,AV_ROUND_UP);
                    size_t before=audioBuf.size();
                    audioBuf.resize(before+(size_t)mx*cfg::AUDIO_CH);
                    float* out=audioBuf.data()+before;
                    int n=swr_convert(swr,(uint8_t**)&out,mx,(const uint8_t**)f->data,f->nb_samples);
                    audioBuf.resize(before+(size_t)n*cfg::AUDIO_CH);
                }
            }
            av_packet_unref(p);
        }
        av_packet_free(&p); av_frame_free(&f);
        // Seek video stream back to start so nextVideoFrame() works from frame 0
        av_seek_frame(fmt,-1,0,AVSEEK_FLAG_BACKWARD);
        if (vctx) avcodec_flush_buffers(vctx);
        if (actx) avcodec_flush_buffers(actx);
        eof=false;
        std::cerr<<"[audio] preDecodeAudio: "<<audioBuf.size()<<" samples\n";
    }

    bool open(const std::string& path, bool wantAudio=true) {
        if (avformat_open_input(&fmt,path.c_str(),nullptr,nullptr)<0) {
            std::cerr<<"[video] Cannot open: "<<path<<"\n"; return false;
        }
        avformat_find_stream_info(fmt,nullptr);
        for (unsigned i=0;i<fmt->nb_streams;++i) {
            auto t=fmt->streams[i]->codecpar->codec_type;
            if (t==AVMEDIA_TYPE_VIDEO&&vsi<0) vsi=(int)i;
            if (t==AVMEDIA_TYPE_AUDIO&&asi<0) asi=(int)i;
        }
        if (vsi>=0) {
            const AVCodec* vc=avcodec_find_decoder(fmt->streams[vsi]->codecpar->codec_id);
            vctx=avcodec_alloc_context3(vc);
            avcodec_parameters_to_context(vctx,fmt->streams[vsi]->codecpar);
            avcodec_open2(vctx,vc,nullptr);
            w=vctx->width; h=vctx->height;
            sws=sws_getContext(w,h,vctx->pix_fmt,w,h,AV_PIX_FMT_RGB24,
                               SWS_BILINEAR,nullptr,nullptr,nullptr);
        }
        if (wantAudio&&asi>=0) {
            const AVCodec* ac=avcodec_find_decoder(fmt->streams[asi]->codecpar->codec_id);
            actx=avcodec_alloc_context3(ac);
            avcodec_parameters_to_context(actx,fmt->streams[asi]->codecpar);
            avcodec_open2(actx,ac,nullptr);
            AVChannelLayout stereo=AV_CHANNEL_LAYOUT_STEREO;
            swr_alloc_set_opts2(&swr,&stereo,AV_SAMPLE_FMT_FLT,cfg::AUDIO_RATE,
                                &actx->ch_layout,actx->sample_fmt,actx->sample_rate,0,nullptr);
            swr_init(swr);
        }
        pkt=av_packet_alloc(); frm=av_frame_alloc();
        return true;
    }

    bool nextVideoFrame(uint8_t* dst) {
        while (!eof) {
            int ret=av_read_frame(fmt,pkt);
            if (ret<0){eof=true;break;}
            if (pkt->stream_index==vsi&&vctx) {
                avcodec_send_packet(vctx,pkt);
                ret=avcodec_receive_frame(vctx,frm);
                av_packet_unref(pkt);
                if (ret==0) {
                    uint8_t* dp[1]={dst}; int ls[1]={w*3};
                    sws_scale(sws,frm->data,frm->linesize,0,h,dp,ls);
                    return true;
                }
            } else { av_packet_unref(pkt); }
        }
        return false;
    }

    void close() {
        if (sws)  sws_freeContext(sws);
        if (swr)  swr_free(&swr);
        if (vctx) avcodec_free_context(&vctx);
        if (actx) avcodec_free_context(&actx);
        if (pkt)  av_packet_free(&pkt);
        if (frm)  av_frame_free(&frm);
        if (fmt)  avformat_close_input(&fmt);
    }
    ~VideoDecoder(){ close(); }
};

// ============================================================================
// PNG SPRITE SEQUENCE
// ============================================================================
struct SpriteSeq {
    std::vector<GLuint> frames;
    int  cur=0;
    bool done=false, loaded=false;

    bool load(const std::string& dir) {
        if (!fs::exists(dir)){std::cerr<<"[sprite] Dir not found: "<<dir<<"\n";return false;}
        std::vector<fs::path> paths;
        for (auto& e:fs::directory_iterator(dir)) {
            auto fn=e.path().filename().string();
            if (e.path().extension()==".png" && fn.substr(0,2)!="._")
                paths.push_back(e.path());
        }
        std::sort(paths.begin(),paths.end());
        if (paths.empty()){std::cerr<<"[sprite] No PNGs in "<<dir<<"\n";return false;}
        frames.reserve(paths.size());
        for (auto& p:paths) {
            int w,h,ch;
            unsigned char* data=stbi_load(p.string().c_str(),&w,&h,&ch,STBI_rgb_alpha);
            if (!data){std::cerr<<"[sprite] Failed: "<<p<<"\n";continue;}
            GLuint t; glGenTextures(1,&t);
            glBindTexture(GL_TEXTURE_2D,t);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,data);
            stbi_image_free(data);
            frames.push_back(t);
        }
        cur=0; done=false; loaded=!frames.empty();
        return loaded;
    }

    void reset(){ cur=0; done=false; }

    GLuint advance() {
        if (frames.empty()){done=true;return 0;}
        GLuint t=frames[cur];
        if (cur<(int)frames.size()-1) ++cur; else done=true;
        return t;
    }

    GLuint current()    const{ return frames.empty()?0:frames[cur]; }
    GLuint firstFrame() const{ return frames.empty()?0:frames.front(); }
    GLuint lastFrame()  const{ return frames.empty()?0:frames.back(); }
};

// ============================================================================
// AUDIO MIXER
// ============================================================================
struct AudioMixer {
    ma_device device{};
    VideoDecoder*        vidSrc   = nullptr;  // set to introVid during intro, nullptr otherwise
    std::vector<float>*  walkPcm  = nullptr;
    std::atomic<size_t>* walkPos  = nullptr;
    std::atomic<bool>*   walkPlay = nullptr;
    std::atomic<bool>*   walkLoop = nullptr;

    static void callback(ma_device* dev, void* out, const void*, ma_uint32 frames) {
        auto*  self=static_cast<AudioMixer*>(dev->pUserData);
        float* dst =static_cast<float*>(out);
        std::fill(dst,dst+frames*cfg::AUDIO_CH,0.f);
        if (self->vidSrc&&!self->vidSrc->audioBuf.empty()) {
            auto& buf=self->vidSrc->audioBuf;
            size_t pos=self->vidSrc->audioPos.load();
            for (ma_uint32 i=0;i<frames*cfg::AUDIO_CH;++i,++pos) {
                if (pos>=buf.size()) pos=0;
                dst[i]+=buf[pos]*0.8f;
            }
            self->vidSrc->audioPos.store(pos);
        }
        if (self->walkPlay&&self->walkPlay->load()&&self->walkPcm) {
            auto& wb=*self->walkPcm;
            size_t wp=self->walkPos->load();
            for (ma_uint32 i=0;i<frames*cfg::AUDIO_CH&&wp<wb.size();++i,++wp)
                dst[i]+=wb[wp];
            if (wp>=wb.size()) {
                if (self->walkLoop->load()) wp=0;
                else{self->walkPlay->store(false);wp=0;}
            }
            self->walkPos->store(wp);
        }
    }

    bool init() {
        ma_device_config c=ma_device_config_init(ma_device_type_playback);
        c.playback.format=ma_format_f32; c.playback.channels=cfg::AUDIO_CH;
        c.sampleRate=cfg::AUDIO_RATE; c.dataCallback=callback; c.pUserData=this;
        return ma_device_init(nullptr,&c,&device)==MA_SUCCESS
            && ma_device_start(&device)==MA_SUCCESS;
    }
    ~AudioMixer(){ ma_device_uninit(&device); }
};

// ============================================================================
// WALK AUDIO CLIP
// ============================================================================
struct AudioClip {
    std::vector<float>  pcm;
    std::atomic<size_t> pos{0};
    std::atomic<bool>   playing{false};
    std::atomic<bool>   looping{false};

    bool load(const std::string& path) {
        VideoDecoder tmp;
        if (!tmp.open(path,true)) return false;
        tmp.preDecodeAudio();
        pcm=std::move(tmp.audioBuf);
        return !pcm.empty();
    }
    void play(bool loop=false){ pos=0; looping=loop; playing=true; }
    void stop()               { playing=false; pos=0; }
};

// ============================================================================
// MAIN APPLICATION
// ============================================================================
struct App {
    GLFWwindow* win       = nullptr;
    GLuint      prog      = 0;   // textured quad
    GLuint      progText  = 0;   // font atlas
    GLuint      progPlain = 0;   // solid color debug
    Quad        quad;

    TitleScreen title;

    // Intro video
    VideoDecoder introVid;
    GLuint       introTex = 0;
    std::vector<uint8_t> introRgb;
    double lastIntroTime  = 0.0;
    double introFrameTime = 1.0/30.0;

    // Background PNG sequence (looping, replaces loop video)
    SpriteSeq seqBg;
    int       bgCur      = 0;
    double    lastBgTime = 0.0;
    double    bgFrameTime= 1.0/cfg::SPRITE_FPS;
    GLuint    bgTex      = 0;

    // Optional outro overlay — drawn over dimmed background in WALKING_RIGHT
    // Populate renders/outro/ to activate; silent failure if missing.
    SpriteSeq seqOutro;
    int       outroCur   = 0;
    GLuint    outroTex   = 0;

    // Sprite sequences
    SpriteSeq seqBackPose, seqEnterRight;
    SpriteSeq seqLeftFoot, seqRightFoot, seqShakeHead;

    double lastSpriteTime  = 0.0;
    double spriteFrameTime = 1.0/cfg::SPRITE_FPS;

    // Walk state
    AppState  appState = AppState::TITLE;
    WalkSub   walkSub  = WalkSub::IDLE;
    int       stepCount= 0;
    glm::mat4 spriteMatrix = glm::mat4(1.f);

    AudioMixer mixer;
    AudioClip   walkClip;

    // Input — set only by key callback (GLFW_PRESS), never by polling
    bool anyKeyPressed   = false;
    bool rightKeyPressed = false;

    // ── Helpers ──────────────────────────────────────────────────────────────
    void updateSpriteMatrix() { spriteMatrix=cycleTransform(stepCount/2); }

    // Used for hold-to-walk: polls directly rather than relying on callback flag
    bool rightHeld() const { return glfwGetKey(win,GLFW_KEY_RIGHT)==GLFW_PRESS; }

    void stepWalk() {
        if (stepCount>=cfg::MAX_CYCLES*2) {
            seqShakeHead.reset(); walkSub=WalkSub::SHAKE_HEAD; return;
        }
        updateSpriteMatrix();
        if ((stepCount%2)==0) { seqLeftFoot.reset();  walkSub=WalkSub::LEFT_FOOT; }
        else                  { seqRightFoot.reset(); walkSub=WalkSub::RIGHT_FOOT; }
        if (!walkClip.playing) walkClip.play(false);
        else walkClip.looping=true;
    }

    void onStepComplete() {
        if (walkSub==WalkSub::SHAKE_HEAD) { walkSub=WalkSub::IDLE; return; }
        ++stepCount;
        walkSub=WalkSub::IDLE;
        if (rightHeld()) stepWalk();
        else             walkClip.stop();
    }

    // ── Key callback ─────────────────────────────────────────────────────────
    static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
        if (action!=GLFW_PRESS) return;
        auto* a=static_cast<App*>(glfwGetWindowUserPointer(w));
        if (!a) return;
        if (key==GLFW_KEY_ESCAPE){ glfwSetWindowShouldClose(w,GLFW_TRUE); return; }
        a->anyKeyPressed   = true;
        a->rightKeyPressed = (key==GLFW_KEY_RIGHT);
    }

    // ── Upload helpers ────────────────────────────────────────────────────────
    void uploadIntroFrame() {
        if (!introVid.nextVideoFrame(introRgb.data())) return;
        glBindTexture(GL_TEXTURE_2D,introTex);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,introVid.w,introVid.h,
                        GL_RGB,GL_UNSIGNED_BYTE,introRgb.data());
    }

    void tickBg() {
        if (seqBg.frames.empty()) return;
        if (bgCur>=(int)seqBg.frames.size()) bgCur=0;
        bgTex=seqBg.frames[bgCur++];
    }

    void tickOutro() {
        if (seqOutro.frames.empty()) return;
        if (outroCur>=(int)seqOutro.frames.size()) outroCur=0;
        outroTex=seqOutro.frames[outroCur++];
    }

    // ── Init ─────────────────────────────────────────────────────────────────
    bool init() {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
        win=glfwCreateWindow(cfg::WIN_W,cfg::WIN_H,"",nullptr,nullptr);
        if (!win) return false;
        glfwMakeContextCurrent(win);
        glfwSwapInterval(0);
        glfwSetWindowUserPointer(win,this);
        glfwSetKeyCallback(win,keyCallback);
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return false;
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

        prog      = makeProgram(VS_SRC,   FS_SRC);
        progText  = makeProgram(VS_SRC,   FS_TEXT);
        progPlain = makeProgram(VS_PLAIN, FS_PLAIN);
        quad.init();

        title.load(progText);

        if (!introVid.open("vids/splice/intro.mp4",true)) return false;
        introTex=makeTex(introVid.w,introVid.h);
        introRgb.resize((size_t)introVid.w*introVid.h*3);
        if (introVid.vsi>=0) {
            AVRational r=introVid.fmt->streams[introVid.vsi]->avg_frame_rate;
            if (r.num>0) introFrameTime=(double)r.den/r.num;
        }
        introVid.preDecodeAudio();

        if (!seqBg.load("renders/background_waves_staggered"))
            std::cerr<<"[bg] No frames in renders/background_waves_staggered\n";

        seqOutro.load("renders/outro");  // optional — silent failure if missing/empty

        seqBackPose.load("renders/back_pose");
        seqEnterRight.load("renders/enter_right");
        seqLeftFoot.load("renders/walk_right/left_foot");
        seqRightFoot.load("renders/walk_right/right_foot");
        seqShakeHead.load("renders/walk_right/shake_head");

        if (!walkClip.load("audios/walking.m4a"))
            std::cerr<<"[audio] walking.m4a not found\n";

        mixer.vidSrc   = nullptr;        // silent during title screen
        mixer.walkPcm  = &walkClip.pcm;
        mixer.walkPos  = &walkClip.pos;
        mixer.walkPlay = &walkClip.playing;
        mixer.walkLoop = &walkClip.looping;
        if (!mixer.init()) std::cerr<<"[audio] mixer init failed\n";

        double t0=glfwGetTime();
        lastIntroTime=lastBgTime=lastSpriteTime=t0;
        title.lastCharTime=title.cursorTimer=t0;
        return true;
    }

    // ── Run ──────────────────────────────────────────────────────────────────
    void run() {
        while (!glfwWindowShouldClose(win)) {
            glfwPollEvents();
            // IMPORTANT: do NOT poll anyKeyPressed/rightKeyPressed here.
            // Held keys from a previous state would bleed through and skip states.
            // rightHeld() is used directly inside onStepComplete() for hold-to-walk.

            double now  = glfwGetTime();
            bool advI   = (now-lastIntroTime)  >= introFrameTime;
            bool advBg  = (now-lastBgTime)     >= bgFrameTime;
            bool advS   = (now-lastSpriteTime) >= spriteFrameTime;
            if (advI)  lastIntroTime  = now;
            if (advBg) { lastBgTime   = now; tickBg(); tickOutro(); }
            if (advS)  lastSpriteTime = now;

            // ── State machine ─────────────────────────────────────────────────
            switch (appState) {

            case AppState::TITLE:
                title.update(now);
                // Only accept key after typing finishes (prevents accidental skip)
                if (anyKeyPressed && title.phase==TitleScreen::Phase::ELLIPSIS) {
                    anyKeyPressed=false; rightKeyPressed=false;
                    introVid.audioPos=0;
                    mixer.vidSrc=&introVid;   // start intro audio
                    uploadIntroFrame();        // prime first frame — no black flash
                    lastIntroTime=now;
                    appState=AppState::INTRO;
                }
                break;

            case AppState::INTRO:
                if (advI) uploadIntroFrame();
                if (introVid.eof || anyKeyPressed) {
                    anyKeyPressed=false; rightKeyPressed=false;
                    mixer.vidSrc=nullptr;     // stop intro audio
                    appState=AppState::BACK_POSE;
                }
                break;

            case AppState::BACK_POSE:
                if (advS) {
                    if (seqBackPose.done) seqBackPose.reset();
                    seqBackPose.advance();
                }
                if (rightKeyPressed) {
                    rightKeyPressed=false;
                    seqEnterRight.reset();
                    appState=AppState::ENTERING_RIGHT;
                }
                break;

            case AppState::ENTERING_RIGHT:
                if (advS) {
                    if (seqEnterRight.loaded) {
                        seqEnterRight.advance();
                        if (seqEnterRight.done) {
                            stepCount=0; updateSpriteMatrix();
                            walkSub=WalkSub::IDLE; appState=AppState::WALKING_RIGHT;
                        }
                    } else {
                        // No enter_right renders — transition immediately
                        stepCount=0; updateSpriteMatrix();
                        walkSub=WalkSub::IDLE; appState=AppState::WALKING_RIGHT;
                    }
                }
                break;

            case AppState::WALKING_RIGHT:
                if (rightKeyPressed) {
                    rightKeyPressed=false;
                    if (walkSub==WalkSub::IDLE) stepWalk();
                }
                if (advS) {
                    switch (walkSub) {
                    case WalkSub::IDLE: break;
                    case WalkSub::LEFT_FOOT:
                        seqLeftFoot.advance();
                        if (seqLeftFoot.done)  onStepComplete(); break;
                    case WalkSub::RIGHT_FOOT:
                        seqRightFoot.advance();
                        if (seqRightFoot.done) onStepComplete(); break;
                    case WalkSub::SHAKE_HEAD:
                        seqShakeHead.advance();
                        if (seqShakeHead.done) onStepComplete(); break;
                    }
                }
                break;
            }

            // ── Render ───────────────────────────────────────────────────────
            glClearColor(0,0,0,1);
            glClear(GL_COLOR_BUFFER_BIT);

            if (appState==AppState::TITLE) {
                // Pure black background, white typed text
                glEnable(GL_BLEND);
                title.render();
            } else {
                // ── Background layer ─────────────────────────────────────────
                glDisable(GL_BLEND);
                if (appState==AppState::INTRO) {
                    drawTex(prog,quad,introTex);
                } else if (appState==AppState::WALKING_RIGHT && seqOutro.loaded) {
                    // Waves at 0.5 opacity, outro at full opacity on top
                    drawTex(prog,quad,bgTex);                           // opaque base
                    glEnable(GL_BLEND);
                    drawTex(prog,quad,bgTex,  glm::mat4(1.f), 0.5f);   // waves dimmed
                    drawTex(prog,quad,outroTex,glm::mat4(1.f), 1.0f);  // outro over
                } else {
                    drawTex(prog,quad,bgTex);
                    glEnable(GL_BLEND);
                }

                // ── Debug placeholder helper ──────────────────────────────────
                auto drawPlaceholder=[&](glm::vec4 color, const glm::mat4& M=glm::mat4(1.f)){
                    glUseProgram(progPlain);
                    glUniformMatrix4fv(glGetUniformLocation(progPlain,"uModel"),1,GL_FALSE,glm::value_ptr(M));
                    glUniform4fv(glGetUniformLocation(progPlain,"uColor"),1,glm::value_ptr(color));
                    quad.draw();
                };

                // ── Sprite layer ──────────────────────────────────────────────
                switch (appState) {
                case AppState::BACK_POSE:
                    if (seqBackPose.loaded)
                        drawTex(prog,quad,seqBackPose.current());
                    else
                        drawPlaceholder({0.2f,0.2f,0.8f,0.4f});  // blue
                    break;

                case AppState::ENTERING_RIGHT:
                    if (seqEnterRight.loaded)
                        drawTex(prog,quad,seqEnterRight.current());
                    else
                        drawPlaceholder({0.2f,0.8f,0.2f,0.4f});  // green
                    break;

                case AppState::WALKING_RIGHT: {
                    GLuint tx=0;
                    switch (walkSub) {
                    case WalkSub::IDLE:
                        // Hold last frame of the most recently completed sequence
                        if      (stepCount==0)                 tx=seqEnterRight.lastFrame();
                        else if (stepCount>=cfg::MAX_CYCLES*2) tx=seqShakeHead.lastFrame();
                        else if ((stepCount%2)==0)             tx=seqRightFoot.lastFrame();
                        else                                   tx=seqRightFoot.firstFrame();
                        break;
                    case WalkSub::LEFT_FOOT:   tx=seqLeftFoot.current();  break;
                    case WalkSub::RIGHT_FOOT:  tx=seqRightFoot.current(); break;
                    case WalkSub::SHAKE_HEAD:  tx=seqShakeHead.current(); break;
                    }
                    if (tx) drawTex(prog,quad,tx,spriteMatrix);
                    else    drawPlaceholder({0.8f,0.5f,0.1f,0.5f},spriteMatrix);  // orange
                    break;
                }
                default: break;
                }
            }

            glfwSwapBuffers(win);
        }
    }

    ~App() {
        introVid.close();
        glfwDestroyWindow(win);
        glfwTerminate();
    }
};

int main() {
    App app;
    if (!app.init()) { std::cerr<<"Init failed.\n"; return 1; }
    app.run();
    return 0;
}