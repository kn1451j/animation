// ============================================================================
// 2.5D Interactive Animation — main.cpp
// ============================================================================
// Build (macOS):
//   clang -c glad/src/glad.c -Iglad/include -o glad/glad.o   (once)
//   clang++ -std=c++17 main.cpp glad/glad.o -o anim -Iglad/include \
//     -I/opt/homebrew/include -L/opt/homebrew/lib \
//     -lglfw -lavcodec -lavformat -lavutil -lswscale -lswresample \
//     -framework OpenGL
// Build (Windows): use CMakeLists.txt with vcpkg + pre-built FFmpeg
//
// Run from the repo root — every asset path is relative. Two of the sprite
// directories are generated: run `python3 audience_split.py` (level-3 audience
// split into the layers the bug is drawn between) and `python3
// exit_screen_assets.py` (the exit screen's tree sphere and rim) once.

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
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// CONFIG
// ============================================================================
namespace cfg {
    constexpr int    WIN_W        = 1920;
    constexpr int    WIN_H        = 1080;
    constexpr double SPRITE_FPS   = 60.0;
    constexpr int    MAX_CYCLES   = 6;
    constexpr int    AUDIO_RATE   = 44100;
    constexpr int    AUDIO_CH     = 2;
    constexpr float  FONT_SIZE_PX = 32.f;   // title prompt — small mac-terminal scale
    constexpr int    FONT_ATLAS   = 512;
    // macOS Terminal "Grass" preset palette
    constexpr float  GRASS_BG_R   = 0.094f;
    constexpr float  GRASS_BG_G   = 0.384f;
    constexpr float  GRASS_BG_B   = 0.094f;
    constexpr float  GRASS_FG_R   = 0.93f;
    constexpr float  GRASS_FG_G   = 0.86f;
    constexpr float  GRASS_FG_B   = 0.55f;
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

// namespace hom {
//     static constexpr float H[3][3] = {
//         { 0.926354f,  0.000925f, 311.808373f },
//         { 0.017365f,  1.066689f, -38.550526f },
//         { 0.00003f,   0.0f,        1.0f      }
//     };
//     static constexpr float H_right[3][3] = 
//     {{  0.820061,  -0.015444, -44.729205},
//     { -0.028322,   1.004358,  -6.78747 },
//     { -0.000028,  -0.000002,   1.0f      }};
// }

namespace hom {
    // Step homographies derived directly from rendered frames via 4-corner
    // alpha-bbox mapping. H_right_step: walk_right/left_foot/0001 →
    // walk_right/right_foot/0167. H_left_step: walk_left/left_foot/0089 →
    // walk_left/right_foot/0242, expressed in post-(H_left_px_to_right) space
    // since Hg_pixel acts in that space. The *_half variants are matrix
    // square roots — turn-time correction for mid-pair direction switches.
    // H_right_base offsets the sprite toward the left edge of the screen.
    static constexpr float H_right_base[3][3] = {
        { 1.f, 0.f, -200.f },
        { 0.f, 1.f,    0.f },
        { 0.f, 0.f,    1.f }
    };
    // Pure-translation step homographies (scale stripped) so the sprite
    // doesn't grow/shrink across cycles. X = centroid delta, Y = bbox-bottom
    // delta over one cycle so the character's feet land at the right place
    // when crossing a cycle boundary.
    //   walk_right cycle: cx 364.63 → 616.61 (Δx +251.98), bot 902 → 943 (Δy +41)
    //   walk_left  cycle: cx 1510.99 → 1251.67 (Δx -259.32), bot 895 → 918 (Δy +23)
    static constexpr float H_right_step[3][3] = {
        { 1.f, 0.f, 251.980f },
        { 0.f, 1.f,  41.000f },
        { 0.f, 0.f,   1.f    }
    };
    static constexpr float H_right_step_half[3][3] = {
        { 1.f, 0.f, 125.990f },
        { 0.f, 1.f,  20.500f },
        { 0.f, 0.f,   1.f    }
    };
    static constexpr float H_left_step[3][3] = {
        { 1.f, 0.f, -259.320f },
        { 0.f, 1.f,   23.000f },
        { 0.f, 0.f,    1.f    }
    };
    static constexpr float H_left_step_half[3][3] = {
        { 1.f, 0.f, -129.660f },
        { 0.f, 1.f,   11.500f },
        { 0.f, 0.f,    1.f    }
    };
    // Pure translation aligning the alpha centroid of a walk_left frame
    // with the alpha centroid of a walk_right frame.
    static constexpr float H_left_px_to_right[3][3] = {
        { 1.f, 0.f, -1146.34f },
        { 0.f, 1.f,     9.13f },
        { 0.f, 0.f,     1.f   }
    };
}

// Alignment landmarks per frame: centroid X (alpha-weighted) and bbox-bottom Y
// (so character FEET line up across frames, per the asset designer's intent).
namespace align {
    // First frame of each turn animation.
    static constexpr float TURN_RIGHT_FIRST_X = 370.89f,  TURN_RIGHT_FIRST_Y = 830.f;     // walk_right/turn_right/0055
    static constexpr float TURN_LEFT_FIRST_X  = 1528.51f, TURN_LEFT_FIRST_Y  = 878.f;     // walk_left/turn_left/0055
    // Last frame of each turn animation.
    static constexpr float TURN_RIGHT_LAST_X  = 386.11f,  TURN_RIGHT_LAST_Y  = 851.f;     // walk_right/turn_right/0089
    static constexpr float TURN_LEFT_LAST_X   = 1510.99f, TURN_LEFT_LAST_Y   = 895.f;     // walk_left/turn_left/0089

    // IDLE-frame centroids/bottoms per direction × idle-rendering case.
    static constexpr float R_IDLE_FIRST_X = 364.63f,  R_IDLE_FIRST_Y = 902.f;     // walk_right/left_foot/0001 (pre-step)
    static constexpr float R_IDLE_ODD_X   = 504.02f,  R_IDLE_ODD_Y   = 923.f;     // walk_right/left_foot/0087 (after LEFT_FOOT)
    static constexpr float R_IDLE_EVEN_X  = 616.61f,  R_IDLE_EVEN_Y  = 943.f;     // walk_right/right_foot/0167 (after RIGHT_FOOT)
    static constexpr float L_IDLE_FIRST_X = 1510.99f, L_IDLE_FIRST_Y = 895.f;     // walk_left/left_foot/0089
    static constexpr float L_IDLE_ODD_X   = 1371.06f, L_IDLE_ODD_Y   = 899.f;     // walk_left/left_foot/0162
    static constexpr float L_IDLE_EVEN_X  = 1251.67f, L_IDLE_EVEN_Y  = 918.f;     // walk_left/right_foot/0242
}

// Read a PNG and compute its alpha-weighted centroid X plus bbox-bottom Y
// (max y where alpha > threshold). Matches the convention used in the align::
// namespace constants. Returns false on load failure.
static bool computeAlphaCentroidBottom(const std::string& pngPath,
                                       float& cx, float& by) {
    int w, h, ch;
    unsigned char* data = stbi_load(pngPath.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!data) return false;
    double sumA = 0.0, sumAx = 0.0;
    int    maxY = 0;
    const int thresh = 16;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char a = data[(y*w + x)*4 + 3];
            if (a > thresh) {
                sumA  += a;
                sumAx += (double)a * x;
                if (y > maxY) maxY = y;
            }
        }
    }
    stbi_image_free(data);
    if (sumA <= 0.0) return false;
    cx = (float)(sumAx / sumA);
    by = (float)maxY;
    return true;
}

// Alpha bounding box of a PNG, in texels. Needed wherever art has to be placed
// by an edge rather than by its centroid — the exit screen stands the man on a
// ground line and centres the exit animation on the sphere.
static bool computeAlphaBBox(const std::string& pngPath,
                             float& x0, float& y0, float& x1, float& y1) {
    int w, h, ch;
    unsigned char* data = stbi_load(pngPath.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!data) return false;
    int minX = w, minY = h, maxX = -1, maxY = -1;
    const int thresh = 16;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (data[(y*w + x)*4 + 3] > thresh) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
    stbi_image_free(data);
    if (maxX < 0) return false;
    x0 = (float)minX; y0 = (float)minY; x1 = (float)maxX; y1 = (float)maxY;
    return true;
}

// Lexicographically-last .png in a directory (skipping macOS AppleDouble
// files), matching how SpriteSeq::load orders its frames — so this names the
// same image the sequence ends on.
static std::string lastPngIn(const std::string& dir) {
    std::string best;
    if (!fs::exists(dir)) return best;
    for (auto& e : fs::directory_iterator(dir)) {
        auto fn = e.path().filename().string();
        if (e.path().extension() != ".png" || fn.substr(0,2) == "._") continue;
        if (best.empty() || fn > fs::path(best).filename().string())
            best = e.path().string();
    }
    return best;
}

// Single RGBA texture from a PNG, same filtering as SpriteSeq's frames.
// outW/outH report the texel size, which the boiled stills need for placement
// the same way SpriteSeq's texW/texH serve the sequences.
static GLuint loadTexture(const std::string& path, int* outW=nullptr, int* outH=nullptr) {
    int w, h, ch;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!data) { std::cerr<<"[tex] Failed: "<<path<<"\n"; return 0; }
    if (outW) *outW = w;
    if (outH) *outH = h;
    GLuint t; glGenTextures(1,&t);
    glBindTexture(GL_TEXTURE_2D,t);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,w,h,0,GL_RGBA,GL_UNSIGNED_BYTE,data);
    stbi_image_free(data);
    return t;
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

// Place a texture of any size into a screen-space rect. The quad's UV space
// always spans the whole texture, and homPixToNDC treats that span as the full
// WIN_W×WIN_H canvas — so scaling by (w/WIN_W, h/WIN_H) maps the texture onto a
// w×h rect whose top-left texel lands at (x, y) in screen pixels. The
// hand-drawn man and audience sequences are all much smaller than the window
// and are placed with this; the 1920×1080 walk sequences don't need it.
static glm::mat4 rectMatrix(float x, float y, float w, float h) {
    float Hp[3][3] = { { w/(float)cfg::WIN_W, 0.f,                 x },
                       { 0.f,                 h/(float)cfg::WIN_H, y },
                       { 0.f,                 0.f,                 1.f } };
    float Hn[3][3]; homPixToNDC(Hp, Hn);
    return embedHom(Hn);
}

// ============================================================================
// APP STATE
// ============================================================================
// After CENTER_HOLD the piece keeps pulling back: the digital sprite is swapped
// for a hand-drawn one (HAND_DRAWN), then three camera zoom-outs reveal a
// hand-drawn audience one nested level at a time (AUDIENCE_1..3). DOWN advances.
enum class AppState { TITLE, INTRO, BACK_POSE, ENTERING_RIGHT,
                      WALKING_RIGHT, WALKING_LEFT,
                      CENTERING, CENTER_HOLD,
                      HAND_DRAWN, AUDIENCE_1, AUDIENCE_2, AUDIENCE_3,
                      OUTRO };
enum class WalkSub  { IDLE, LEFT_FOOT, RIGHT_FOOT, SHAKE_HEAD, TURN};

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

// Overlay fragment shader: amplify the texture's alpha by uAlpha, but keep
// fully-transparent regions transparent. uAlpha=0.5 → half visibility; =1.0
// → natural intrinsic α; =2.0 → wave regions saturate to fully opaque while
// the parts of the PNG that were fully transparent still let the bg show.
static const char* FS_OVERLAY = R"glsl(
#version 330 core
in vec2 vUV; out vec4 frag;
uniform sampler2D uTex; uniform float uAlpha;
void main(){
    vec4 c = texture(uTex, vUV);
    float a = c.a > 0.02 ? min(1.0, c.a * uAlpha) : 0.0;
    frag = vec4(c.rgb, a);
})glsl";

// Hand-drawn "boil": jitter the UV lookup with low-frequency value noise so the
// Blender sprite's clean vector lines wobble like ink. uBoilSeed is quantised to
// BOIL_HZ so the drawing re-settles in discrete steps instead of sliding
// smoothly — that stepping is what reads as hand-drawn. Amplitude is calibrated
// against renders/man_noise: p90 line displacement there is ~1 px, max ~3 px on
// a 522 px figure. 24×14 noise cells across the canvas ≈ 60 px cells on screen,
// so the figure spans ~6 cells and bends coherently rather than dissolving.
static const char* FS_BOIL = R"glsl(
#version 330 core
in vec2 vUV; out vec4 frag;
uniform sampler2D uTex; uniform float uAlpha;
uniform vec2  uBoilAmp;
uniform float uBoilSeed;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float vnoise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    f=f*f*(3.0-2.0*f);
    return mix(mix(hash(i),           hash(i+vec2(1,0)), f.x),
               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y);
}
void main(){
    vec2 q = vUV * vec2(24.0, 14.0) + uBoilSeed;
    vec2 d = (vec2(vnoise(q), vnoise(q + 37.0)) - 0.5) * 2.0 * uBoilAmp;
    frag = texture(uTex, vUV + d);
    frag.a *= uAlpha;
})glsl";

// Flat-tint line art that also boils. The man and the three audience levels
// used to ship as 100-frame loops, but those frames were one drawing being
// redrawn — ink sat a p90 of 1 px from frame 0 — so the wobble is generated
// here instead and the assets are single stills (see boil_stills.py).
//
// Unlike FS_BOIL, the noise cell count is a uniform rather than a baked 24x14:
// that constant assumed a full 1920x1080 canvas, and these stills are tight
// crops crossing a range of on-screen sizes. drawBoiledTinted derives it from
// the rect so cells stay near the calibrated ~60 screen px however big the
// drawing lands. Noise helpers mirror FS_BOIL's — GLSL has no include.
static const char* FS_BOIL_TINT = R"glsl(
#version 330 core
in vec2 vUV; out vec4 frag;
uniform sampler2D uTex; uniform float uAlpha; uniform vec3 uTint;
uniform vec2  uBoilAmp;
uniform vec2  uBoilCells;
uniform float uBoilSeed;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
float vnoise(vec2 p){
    vec2 i=floor(p), f=fract(p);
    f=f*f*(3.0-2.0*f);
    return mix(mix(hash(i),           hash(i+vec2(1,0)), f.x),
               mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y);
}
void main(){
    vec2 q = vUV * uBoilCells + uBoilSeed;
    vec2 d = (vec2(vnoise(q), vnoise(q + 37.0)) - 0.5) * 2.0 * uBoilAmp;
    frag = vec4(uTint, texture(uTex, vUV + d).a * uAlpha);
})glsl";

// Flat-tint line art: keep the PNG's alpha (which carries the antialiased
// stroke) but replace its colour outright. The hand-drawn art ships as black
// ink, which vanishes against the black backdrop these phases run on, so the
// audience is recoloured to a white outline at draw time rather than in the
// assets.
static const char* FS_TINT = R"glsl(
#version 330 core
in vec2 vUV; out vec4 frag;
uniform sampler2D uTex; uniform float uAlpha; uniform vec3 uTint;
void main(){ frag = vec4(uTint, texture(uTex, vUV).a * uAlpha); })glsl";

// Font atlas — red channel drives alpha, glyphs tinted by uColor (default white)
static const char* FS_TEXT = R"glsl(
#version 330 core
in vec2 vUV; out vec4 frag;
uniform sampler2D uTex; uniform float uAlpha;
uniform vec3 uColor;
void main(){
    float a = texture(uTex, vUV).r;
    frag = vec4(uColor, a * uAlpha);
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

// drawTex through the flat-tint shader — the texture supplies the shape, uTint
// supplies the colour.
static void drawTinted(GLuint prog, Quad& q, GLuint tex, const glm::mat4& M,
                       const glm::vec3& tint, float alpha=1.f) {
    if (!tex) return;
    glUseProgram(prog);
    glUniform3fv(glGetUniformLocation(prog,"uTint"),1,glm::value_ptr(tint));
    drawTex(prog, q, tex, M, alpha);
}

// drawTex through the boil shader. `spriteScale` is the sprite matrix's X
// scale, needed to convert the desired screen-pixel wobble into UV units, and
// `seed` should already be quantised to the boil cadence.
static void drawBoiled(GLuint prog, Quad& q, GLuint tex, const glm::mat4& M,
                       float boilPx, float spriteScale, float seed,
                       float alpha=1.f) {
    if (!tex) return;
    if (spriteScale <= 0.f) spriteScale = 1.f;
    glUseProgram(prog);
    glUniform2f(glGetUniformLocation(prog,"uBoilAmp"),
                boilPx/((float)cfg::WIN_W*spriteScale),
                boilPx/((float)cfg::WIN_H*spriteScale));
    glUniform1f(glGetUniformLocation(prog,"uBoilSeed"), seed);
    drawTex(prog, q, tex, M, alpha);
}

// drawTinted through the boil shader, for the rect-placed hand-drawn stills.
// rectW/rectH are the drawing's on-screen size in pixels, which is all the
// conversion needs here: a UV offset of d shifts the sampled point by d*rectW
// screen px, so the amplitude is just boilPx over the rect. (drawBoiled instead
// works in sprite-matrix units, where the quad always spans the full window.)
// Cells are sized off the same rect to hold the calibrated ~60 px cell.
static void drawBoiledTinted(GLuint prog, Quad& q, GLuint tex, const glm::mat4& M,
                             float rectW, float rectH, float boilPx, float seed,
                             const glm::vec3& tint, float alpha=1.f) {
    if (!tex || rectW <= 0.f || rectH <= 0.f) return;
    glUseProgram(prog);
    glUniform3fv(glGetUniformLocation(prog,"uTint"),1,glm::value_ptr(tint));
    glUniform2f(glGetUniformLocation(prog,"uBoilAmp"), boilPx/rectW, boilPx/rectH);
    glUniform2f(glGetUniformLocation(prog,"uBoilCells"),
                std::max(2.f, rectW/60.f), std::max(2.f, rectH/60.f));
    glUniform1f(glGetUniformLocation(prog,"uBoilSeed"), seed);
    drawTex(prog, q, tex, M, alpha);
}

// A hand-drawn still that boils at runtime, standing in for what used to be a
// 100-frame PNG loop. w/h mirror SpriteSeq's texW/texH so every placement
// expression that sized itself off the sequence still reads the same.
struct BoilStill {
    GLuint tex = 0;
    int    w = 0, h = 0;
    bool   loaded = false;

    bool load(const std::string& path) {
        tex    = loadTexture(path, &w, &h);
        loaded = tex != 0;
        return loaded;
    }
    void unload() {
        if (tex) glDeleteTextures(1, &tex);
        tex = 0; loaded = false;
    }
    GLuint current() const { return tex; }
};

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

    // Candidate prompts — one is chosen at random in load() and stored in
    // `word`. Picked once per process; lasts the lifetime of the title screen.
    std::string word;

    bool load(GLuint prog) {
        static const char* const kCandidates[] = {
            R"(when i was little, i would have these dreams about my family selling drugs. we lived in a
            little room. there was a bathtub in the corner, and it was always filled with needles. one night, 
            i dreamt that i came home to that shack, and my family was inside, dead. 
            i walked out, barefoot and dirty, stepping down the sidewalk.
            i had woken up with a popping wetness in my hands. 
            in the dream, i had stepped on a needle and died.)",
             "..."
        };
        constexpr int kCandidateCount = (int)(sizeof(kCandidates)/sizeof(kCandidates[0]));
        std::mt19937 rng{ std::random_device{}() };
        std::uniform_int_distribution<int> d(0, kCandidateCount - 1);
        word = kCandidates[d(rng)];
        return loadFont(prog);
    }

    bool loadFont(GLuint prog) {
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
        const int wordLen = (int)word.size();
        if (phase == Phase::TYPING) {
            if (charsShown < wordLen && now-lastCharTime >= CHAR_DELAY) {
                ++charsShown; lastCharTime=now;
                if (charsShown==wordLen) { phase=Phase::ELLIPSIS; dotCount=1; lastDotTime=now; }
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
        const std::string fullText = std::string("$ ") + word;
        const int visibleLen = std::min((int)fullText.size(), 2 + charsShown);
        std::string suffix;
        if (phase==Phase::ELLIPSIS) suffix = std::string(dotCount,'.');
        else if (cursorOn)          suffix = "_";

        const float W=float(cfg::WIN_W), H=float(cfg::WIN_H);
        auto ndcX=[&](float px){ return (px/W)*2.f-1.f; };
        auto ndcY=[&](float py){ return 1.f-(py/H)*2.f; };

        // Top-left anchor with a small margin — mac-terminal prompt style.
        const float sx = 32.f;
        const float sy = cfg::FONT_SIZE_PX + 24.f;
        const float rightMargin = W - 32.f;
        const float lineHeight  = cfg::FONT_SIZE_PX * 1.3f;

        // Pre-compute the start position (cx, cy) for each char in fullText
        // with word-wrap based on the FULL text. Doing wrap on the full text
        // keeps positions stable during the typewriter reveal.
        std::vector<float> px(fullText.size()+1), py(fullText.size()+1);
        {
            // Cache the space's advance width once.
            float tx=0, ty=0;
            stbtt_aligned_quad sq;
            stbtt_GetBakedQuad(cdata,cfg::FONT_ATLAS,cfg::FONT_ATLAS,
                               ' '-32,&tx,&ty,&sq,1);
            const float spaceW = tx;

            auto measureWordFrom = [&](size_t start) {
                float ax=0, ay=0;
                for (size_t k=start; k<fullText.size(); ++k) {
                    char c = fullText[k];
                    if (c==' '||c=='\n') break;
                    unsigned char uc=(unsigned char)c;
                    if (uc<32||uc>=128) continue;
                    stbtt_aligned_quad q;
                    stbtt_GetBakedQuad(cdata,cfg::FONT_ATLAS,cfg::FONT_ATLAS,
                                       uc-32,&ax,&ay,&q,1);
                }
                return ax;
            };

            float cx=sx, cy=sy;
            for (size_t i=0; i<fullText.size(); ++i) {
                px[i]=cx; py[i]=cy;
                char c = fullText[i];
                if (c=='\n') { cx=sx; cy+=lineHeight; continue; }
                if (c==' ') {
                    // Skip any extra leading whitespace from raw-string indent.
                    if (cx==sx) continue;
                    const float wordW = measureWordFrom(i+1);
                    if (cx + spaceW + wordW > rightMargin) {
                        // Wrap: drop this space, advance to next line.
                        cx=sx; cy+=lineHeight; continue;
                    }
                }
                unsigned char uc=(unsigned char)c;
                if (uc<32||uc>=128) continue;
                stbtt_aligned_quad bq;
                stbtt_GetBakedQuad(cdata,cfg::FONT_ATLAS,cfg::FONT_ATLAS,
                                   uc-32,&cx,&cy,&bq,1);
            }
            px[fullText.size()]=cx; py[fullText.size()]=cy;
        }

        struct Vert { float x,y,u,v; };
        std::vector<Vert>     verts;
        std::vector<uint32_t> inds;
        uint32_t vi=0;

        auto emit = [&](unsigned char uc, float& cx, float& cy) {
            if (uc<32||uc>=128) return;
            stbtt_aligned_quad bq;
            stbtt_GetBakedQuad(cdata,cfg::FONT_ATLAS,cfg::FONT_ATLAS,uc-32,&cx,&cy,&bq,1);
            verts.push_back({ndcX(bq.x0),ndcY(bq.y0),bq.s0,bq.t0});
            verts.push_back({ndcX(bq.x1),ndcY(bq.y0),bq.s1,bq.t0});
            verts.push_back({ndcX(bq.x1),ndcY(bq.y1),bq.s1,bq.t1});
            verts.push_back({ndcX(bq.x0),ndcY(bq.y1),bq.s0,bq.t1});
            inds.insert(inds.end(),{vi,vi+1,vi+2,vi+2,vi+3,vi});
            vi+=4;
        };

        // Draw the visible portion of fullText at its precomputed positions.
        for (int i=0; i<visibleLen; ++i) {
            char c = fullText[i];
            if (c=='\n') continue;
            float cx=px[i], cy=py[i];
            emit((unsigned char)c, cx, cy);
        }
        // Suffix (cursor / ellipsis) anchored at the post-visible position.
        if (!suffix.empty()) {
            float cx=px[visibleLen], cy=py[visibleLen];
            for (char c : suffix) emit((unsigned char)c, cx, cy);
        }

        if (verts.empty()) return;
        glBindVertexArray(tvao);
        glBindBuffer(GL_ARRAY_BUFFER,tvbo);
        glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(verts.size()*sizeof(Vert)),verts.data(),GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,tebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)(inds.size()*sizeof(uint32_t)),inds.data(),GL_DYNAMIC_DRAW);
        glUseProgram(progText);
        glm::mat4 I(1.f);
        glUniformMatrix4fv(glGetUniformLocation(progText,"uModel"),1,GL_FALSE,glm::value_ptr(I));
        glUniform1f(glGetUniformLocation(progText,"uAlpha"),1.f);
        glUniform3f(glGetUniformLocation(progText,"uColor"),
                    cfg::GRASS_FG_R, cfg::GRASS_FG_G, cfg::GRASS_FG_B);
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
    bool loopVideo = false;

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

    void seekToStart() {
        av_seek_frame(fmt,-1,0,AVSEEK_FLAG_BACKWARD);
        if (vctx) avcodec_flush_buffers(vctx);
        eof=false;
    }

    bool nextVideoFrame(uint8_t* dst) {
        while (!eof) {
            int ret=av_read_frame(fmt,pkt);
            if (ret<0){
                if (loopVideo){ seekToStart(); continue; }
                eof=true; break;
            }
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
        if (sws)  { sws_freeContext(sws); sws = nullptr; }   // null it: close() may run twice
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
    // Texel dimensions of the frames. The walk/center sequences are all
    // 1920×1080 (so they blit 1:1 through a pixel homography), but the
    // hand-drawn sequences are much smaller and need rectMatrix() to place
    // them, which requires knowing their size.
    int  texW=0, texH=0;

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
            if (frames.empty()){ texW=w; texH=h; }
            frames.push_back(t);
        }
        cur=0; done=false; loaded=!frames.empty();
        return loaded;
    }

    // Release every frame's GL texture. Called once a sequence's phase is
    // behind us — these are 100s of full-frame RGBA images each, so holding
    // onto them for the whole run is the app's dominant memory cost.
    void unload() {
        if (!frames.empty())
            glDeleteTextures((GLsizei)frames.size(), frames.data());
        frames.clear();
        cur=0; done=false; loaded=false;
    }

    void reset(){ cur=0; done=false; }

    GLuint advance() {
        if (frames.empty()){done=true;return 0;}
        GLuint t=frames[cur];
        if (cur<(int)frames.size()-1) ++cur; else done=true;
        return t;
    }

    // Wrap instead of latching `done` — for the hand-drawn sequences, which
    // are short loops that run for as long as their phase is on screen.
    GLuint advanceLoop() {
        if (frames.empty()) return 0;
        GLuint t=frames[cur];
        cur = (cur+1) % (int)frames.size();
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
    VideoDecoder*        vidSrc   = nullptr;  // primary video audio (loops if buf wraps)
    VideoDecoder*        vidSrc2  = nullptr;  // secondary video audio (mixed in)
    std::vector<float>*  walkPcm  = nullptr;
    std::atomic<size_t>* walkPos  = nullptr;
    std::atomic<bool>*   walkPlay = nullptr;
    std::atomic<bool>*   walkLoop = nullptr;
    std::atomic<float>   bgGain{1.0f};        // multiplies the vidSrc audio
    std::atomic<float>   bgGain2{1.0f};       // multiplies vidSrc2 audio

    static void callback(ma_device* dev, void* out, const void*, ma_uint32 frames) {
        auto*  self=static_cast<AudioMixer*>(dev->pUserData);
        float* dst =static_cast<float*>(out);
        std::fill(dst,dst+frames*cfg::AUDIO_CH,0.f);
        if (self->vidSrc&&!self->vidSrc->audioBuf.empty()) {
            auto& buf=self->vidSrc->audioBuf;
            size_t pos=self->vidSrc->audioPos.load();
            const float gain = 0.8f * self->bgGain.load();
            for (ma_uint32 i=0;i<frames*cfg::AUDIO_CH;++i,++pos) {
                if (pos>=buf.size()) pos=0;
                dst[i]+=buf[pos]*gain;
            }
            self->vidSrc->audioPos.store(pos);
        }
        if (self->vidSrc2&&!self->vidSrc2->audioBuf.empty()) {
            auto& buf=self->vidSrc2->audioBuf;
            size_t pos=self->vidSrc2->audioPos.load();
            const float gain = 0.8f * self->bgGain2.load();
            for (ma_uint32 i=0;i<frames*cfg::AUDIO_CH;++i,++pos) {
                if (pos>=buf.size()) pos=0;
                dst[i]+=buf[pos]*gain;
            }
            self->vidSrc2->audioPos.store(pos);
        }
        if (self->walkPlay&&self->walkPlay->load()&&self->walkPcm) {
            auto& wb=*self->walkPcm;
            size_t wp=self->walkPos->load();
            // Softer than video audio (which mixes at 0.8x) so walking sits
            // under the bg audio instead of masking it.
            constexpr float kWalkGain = 0.5f;
            for (ma_uint32 i=0;i<frames*cfg::AUDIO_CH&&wp<wb.size();++i,++wp)
                dst[i]+=wb[wp]*kWalkGain;
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
    GLuint      progOverlay = 0; // textured quad, alpha from uniform only
    GLuint      progText  = 0;   // font atlas
    GLuint      progPlain = 0;   // solid color debug
    GLuint      progBoil  = 0;   // textured quad + hand-drawn line wobble
    GLuint      progBoilTint = 0;// the same wobble, recoloured — hand-drawn stills
    GLuint      progTint  = 0;   // textured quad, alpha kept, colour replaced
    Quad        quad;

    TitleScreen title;

    // Title audio
    VideoDecoder titleAudio;

    // Exit audio — voice recording played during the OUTRO exit animation.
    VideoDecoder exitAudio;

    // Intro video
    VideoDecoder introVid;
    GLuint       introTex = 0;
    std::vector<uint8_t> introRgb;
    double lastIntroTime  = 0.0;
    double introFrameTime = 1.0/30.0;

    // Background video (station.mp4 or grass.mp4, looping)
    VideoDecoder bgVid;
    GLuint       bgVidTex = 0;
    int          bgVidTexW = 0, bgVidTexH = 0;  // current texture dims
    std::vector<uint8_t> bgVidRgb;
    double lastBgVidTime  = 0.0;
    double bgVidFrameTime = 1.0/24.0;

    // Background PNG sequence (waves overlay at 0.5 alpha)
    SpriteSeq seqBg;
    int       bgCur      = 0;
    double    lastBgTime = 0.0;
    double    bgFrameTime= 1.0/cfg::SPRITE_FPS;
    GLuint    bgTex      = 0;

    // Title-screen ant PNG sequence (rendered as a fullscreen background under
    // the terminal text). Source is masked: only ant pixels are non-black, so
    // drawing over a black clear leaves the rest of the screen black.
    SpriteSeq seqAnts;
    double    lastAntTime  = 0.0;
    double    antFrameTime = 1.0/24.0;  // match source video framerate

    // Scene state. Background scene cycles through kScenes[]; sceneTransition()
    // advances sceneIdx by 1 (mod count). Cycle: station(waves) → grass →
    // bridge(waves) → grass → back to station.
    struct Scene { const char* path; bool showsWaves; bool animatesBg; };
    static constexpr Scene kScenes[4] = {
        {"vids/splice/station.mp4",       true,  false},
        {"vids/splice/grass.mp4",         false, true },
        {"vids/splice/bridge.mp4",        true,  true },
        {"vids/splice/grass.mp4",         false, true },
    };
    int  sceneIdx         = 0;
    bool showWavesOverlay = false;
    bool firstStepTaken   = false;

    // Sprite sequences
    SpriteSeq seqBackPose, seqEnterRight;
    SpriteSeq seqLeftFoot, seqRightFoot, seqShakeHead, seqTurnLeft;
    SpriteSeq seqWLTurnRight, seqWLLeftFoot, seqWLRightFoot, seqWLShakeHead;

    double lastSpriteTime  = 0.0;
    double spriteFrameTime = 1.0/cfg::SPRITE_FPS;

    // Per-direction bundle of sprite sequences + step delta. Lets the main
    // loop drive both WALKING_RIGHT and WALKING_LEFT through one code path.
    struct WalkDir {
        SpriteSeq* leftFoot;
        SpriteSeq* rightFoot;
        SpriteSeq* shakeHead;
        SpriteSeq* turnIn;     // animation shown when turning INTO this direction
        int        stepDelta;  // +1 right, -1 left
    };
    WalkDir dirRight{};
    WalkDir dirLeft{};

    // Walk state
    AppState  appState = AppState::TITLE;
    WalkSub   walkSub  = WalkSub::IDLE;
    int       stepCount= 0;
    // Tracks how many feet have stepped within the current cycle (0, 1, 2).
    // When it reaches 2 at the next startStep, we multiply by the step
    // homography and reset to 0. Persists across direction changes so a
    // mid-pair turn rolls over correctly.
    int       stepsInCycle = 0;
    glm::mat4 spriteMatrix = glm::mat4(1.f);
    // Cumulative pixel-space homography for the sprite. Seeded from
    // hom::H_right_base. Right-walk pair boundaries multiply by H_right_step,
    // left-walk pair boundaries multiply by H_left_step.
    float     Hg_pixel[3][3] = {
        { hom::H_right_base[0][0], hom::H_right_base[0][1], hom::H_right_base[0][2] },
        { hom::H_right_base[1][0], hom::H_right_base[1][1], hom::H_right_base[1][2] },
        { hom::H_right_base[2][0], hom::H_right_base[2][1], hom::H_right_base[2][2] }
    };
    // Turn-only correction. When a direction switch lands mid-pair (one foot
    // stepped, the other didn't), this holds the half-step homography for the
    // outgoing direction so the TURN animation renders at the actual
    // mid-cycle screen position. Folded into Hg_pixel when the turn ends.
    float     Hg_turn[3][3]  = {{1,0,0},{0,1,0},{0,0,1}};
    // Cumulative vertical-lift homography. Accumulated per sprite-frame tick
    // while an animation is actually advancing — never during IDLE — at a
    // rate matched to the bbox-bottom drop of the current animation:
    //   walk_right cycle:  41 px / 167 sprite-frames ≈ 0.245
    //   walk_left  cycle:  23 px / 155 sprite-frames ≈ 0.148
    //   turn_right anim:   21 px / 35  sprite-frames ≈ 0.600
    //   turn_left  anim:   17 px / 35  sprite-frames ≈ 0.486
    float     Hg_lift[3][3]  = {{1,0,0},{0,1,0},{0,0,1}};
    static constexpr float LIFT_WALK_RIGHT = 0.245f;
    static constexpr float LIFT_WALK_LEFT  = 0.148f;
    static constexpr float LIFT_TURN_RIGHT = 0.600f;
    static constexpr float LIFT_TURN_LEFT  = 0.486f;
    // X-axis "send back" for turn_left: TURN_LEFT_FIRST X 1528.51 → TURN_LEFT_LAST
    // X 1510.99 = −17.52 px natural drift over 35 turn frames. Counter-translate
    // the same magnitude in screen-space so the last turn frame's center sits
    // back near the pre-turn idle center. (Y is already cancelled by LIFT_TURN_LEFT.)
    static constexpr float LIFT_TURN_LEFT_X = 17.52f / 35.f;  // ≈ 0.5006 px/frame

    AudioMixer mixer;
    AudioClip   walkClip;

    // Input — set only by key callback (GLFW_PRESS), never by polling
    bool anyKeyPressed   = false;
    bool rightKeyPressed = false;
    bool leftKeyPressed  = false;
    bool downKeyPressed  = false;

    // CENTERING / OUTRO state
    SpriteSeq seqCenter;          // 53-frame transition + idle pose
    SpriteSeq seqSmoothBug;       // 301 RGBA frames — bug clip, played once in OUTRO
    SpriteSeq seqExit;            // exit/ frame sequence — final exit animation
    bool      pendingCenter = false;   // set when DOWN pressed mid-step in WALKING_LEFT
    int       centeringFrame = 0;
    float     T_align_tx = 0.f, T_align_ty = 0.f;
    float     Hg_lift_start_y = 0.f;   // snapshot for lift-decay during CENTERING
    float     centerFirstX = 0.f, centerFirstY = 0.f;  // alpha-centroid X / bbox-bottom Y of center_sprite/0000
    float     centerLastX  = 0.f, centerLastY  = 0.f;  // …of the LAST center_sprite frame (centered pose)
    float     wavesAlpha = 0.5f;       // ramped 0.5 → 2.0 during CENTERING
    // The waves are treated as a backdrop that shares the camera move: they
    // push in across CENTERING, then pull back in proportion with every
    // zoom-out phase. WAVES_CENTER_ZOOM is chosen so that even at the widest
    // shot the scaled texture still covers the frame (1.5 × Z_D/Z_A ≈ 1.04),
    // which is what stops black edges creeping in at the borders.
    float     wavesScale = 1.f;
    static constexpr float WAVES_CENTER_ZOOM = 1.5f;
    // Fade the scene video to black across CENTERING; 0 elsewhere, 1 once
    // centering ends so the background stays black through the OUTRO.
    float     blackAlpha = 0.f;

    // OUTRO: begins on a key press from CENTER_HOLD (waves gone). The bug clip
    // plays full-frame over black with the hand-drawn exit/ animation composited
    // on top; the (quieter) exit voice recording plays underneath. Each clip
    // plays through once then cuts to black. Runs until the recording finishes
    // (outroEndTime), then quits.
    double     lastOutroFrameTime = 0.0;   // bug-clip cadence timer
    double     lastExitFrameTime  = 0.0;   // exit-anim cadence timer
    double     exitFrameSec       = 1.0/24.0;  // set at exit start; see AUDIENCE_3
    double     outroEndTime       = 0.0;
    static constexpr double OUTRO_FRAME_SEC   = 1.0/24.0;  // bug clip ~24 fps
    static constexpr float  EXIT_AUDIO_GAIN   = 0.5f;      // exit recording volume

    // ── Hand-drawn pull-back (HAND_DRAWN → AUDIENCE_1/2/3) ───────────────────
    // Hand-drawn line art on transparent backgrounds. The audience levels are
    // nested bottom-left-aligned crops of one master drawing (l1 sits in l2 at
    // (0,244), l2 sits in l3 at (0,158)), so a single world origin places every
    // one of them. l3 is pre-split by boil_stills.py into the near figures
    // (l3_back) and the receding rows (l3_front) so the bug clip can be drawn
    // between them.
    //
    // These were 100-frame loops until the frames were measured against each
    // other: ink sat a median of 0 px and a p90 of 1 px from frame 0, i.e. one
    // drawing being redrawn rather than anything animating. They are single
    // stills now, boiled by FS_BOIL_TINT at the same 20 fps the loops ran at.
    // Splitting l3 once rather than per frame also stopped ink popping between
    // the two layers, which the per-frame blob labelling used to cause.
    BoilStill stillMan, stillAudL1, stillAudL2, stillAudL3Back, stillAudL3Front;
    static constexpr double HAND_BOIL_HZ = 20.0;   // re-settle rate, was the loop cadence
    static constexpr float  HAND_BOIL_PX = 1.0f;   // p90 line displacement, measured

    // Everything in these phases is placed by one virtual camera that zooms
    // about a focal point on the screen's bottom edge, so the man's feet and
    // the audience's bottom row stay pinned to y = WIN_H for free:
    //     screenX = CAM_FOCAL_X + Z * (worldX - CAM_FOCAL_X)
    //     screenScale = Z * worldScale
    // CAM_FOCAL_X and CAM_ZOOM_STEP are the exact solution to sending the man's
    // centroid 960 → 720 → … → 320 over three equal zoom steps: with
    // a = 960 - F, (720-F) = k(960-F) and (320-F) = k³(960-F) reduce to
    // a² - 2160a + 172800 = 0, giving a = 1080 + sqrt(993600).
    static constexpr float CAM_FOCAL_X    = -1116.796f;
    static constexpr float CAM_ZOOM_STEP  = 0.884434f;
    // Man at Z=1 (the widest shot): centroid X 320, texture scale such that his
    // figure is 1.15× the 367 px the digital sprite occupies at CENTER_HOLD.
    static constexpr float MAN_WORLD_X     = 320.0f;
    static constexpr float MAN_WORLD_SCALE = 0.559300f;
    // Audience at Z=1: drawing's bottom-left corner at x=480, scaled so the
    // full level-3 crop spans 480 → 1920 (1440/1394).
    static constexpr float AUD_WORLD_X     = 480.0f;
    static constexpr float AUD_WORLD_SCALE = 1.033000f;

    // Once the hand-drawn phase starts, the scene video comes back instead of
    // black and a low-opacity dark green wash goes over everything — video,
    // crowd and man alike. It switches on hard at that boundary rather than
    // fading, so the swap to the hand-drawn sprite reads as a cut.
    float  greenAlpha = 0.f;
    static constexpr float GREEN_R = 0.05f, GREEN_G = 0.16f, GREEN_B = 0.09f;
    static constexpr float GREEN_MAX = 0.38f;

    int    stage  = 0;      // 0 = HAND_DRAWN … 3 = AUDIENCE_3
    float  stageT = 1.f;    // 0→1 progress of the move into `stage`
    double stageStartTime = 0.0;
    static constexpr double STAGE_XFADE_SEC = 1.2;   // zoom + cross-fade length
    float  manCentroidX = 0.f, manBottomY = 0.f;     // renders/man_noise anchor

    // Procedural boil on the digital sprite during CENTERING / CENTER_HOLD.
    static constexpr float  BOIL_PX = 2.5f;   // peak wobble, screen pixels
    static constexpr double BOIL_HZ = 12.0;   // re-settle rate

    // The hand-drawn ink is black, which is invisible against the black these
    // phases run on — the audience is recoloured to a white outline at draw
    // time. The man stays black, as he already does at CENTER_HOLD.
    glm::vec3 audTint{1.f, 1.f, 1.f};

    // ── Exit screen ──────────────────────────────────────────────────────────
    // A "vignette sphere" — the backyard-tree drawing from the katias-stuff
    // site, softly faded into a circle and ringed by the circle hand-drawn on
    // the layout sketch — with the exit animation playing over it and the man
    // standing to its right. The pair is centred and fills 3/4 of the screen.
    // Built by exit_screen_assets.py; all three share one circle centre and
    // radius on a square canvas, so one square rect places all of them.
    GLuint texExitDisc = 0, texTreeClean = 0, texTreeNoise = 0, texRim = 0;
    float  manFigX0 = 0.f, manFigY0 = 0.f, manFigX1 = 0.f, manFigY1 = 0.f;
    float  exitBoxX0 = 0.f, exitBoxY0 = 0.f, exitBoxX1 = 0.f, exitBoxY1 = 0.f;
    // The stack — exit text, then the sphere with the man beside it — is
    // centred and half the screen tall. Its parts are fractions of that so the
    // whole thing rescales from one number.
    static constexpr float  EXIT_STACK_H    = 0.50f * (float)cfg::WIN_H;
    static constexpr float  EXIT_ANIM_FRAC  = 0.18f;  // exit-text band / stack
    static constexpr float  EXIT_VGAP_FRAC  = 0.04f;  // text→sphere gap / stack
    static constexpr float  EXIT_MAN_FRAC   = 0.75f;  // man height / sphere Ø
    static constexpr float  EXIT_GAP_FRAC   = 0.05f;  // sphere→man gap / sphere Ø
    // Sphere is a white field with the tree dark on it; everything outside it
    // stays black, and the man beside it stays white.
    glm::vec3 exitInkTint{0.f, 0.f, 0.f};
    // The site's `doodle-boil`: the noise twin flips through four (opacity,
    // sub-pixel offset) states at 8 fps with steps(1), which reads as the line
    // being redrawn rather than cross-fading. Offsets are CSS px, scaled up to
    // stay visible at the sphere's on-screen size.
    static constexpr double EXIT_BOIL_SEC = 0.5 / 4.0;
    static constexpr float  EXIT_BOIL_OFF = 2.0f;
    static constexpr float  kBoilAlpha[4] = { 0.9f,  0.3f,  1.0f, 0.5f };
    static constexpr float  kBoilDX[4]    = { 0.0f,  0.7f, -0.6f, 0.4f };
    static constexpr float  kBoilDY[4]    = { 0.0f, -0.6f,  0.5f, 0.7f };

    // Bug overlay during AUDIENCE_3: rolled once a second, plays through once.
    bool   bugActive = false;
    double lastBugRoll = 0.0;
    float  bugPosX = 0.f, bugPosY = 0.f;   // screen point the bug crawls around
    static constexpr double BUG_ROLL_SEC = 1.0;
    static constexpr float  BUG_PROB     = 0.20f;
    static constexpr float  BUG_SCALE    = 0.45f;   // shrink the 1920×1080 clip
    static constexpr float  BUG_ALPHA    = 0.55f;   // sit under the ink, not on it
    // The motion-isolated clip is a ~236 px disc that wanders over texels
    // x 716–1094, y 447–775 across its 301 frames. Placing by the centre of
    // that travelled box, and insetting by its half-extent, keeps every frame
    // of the bug inside the crowd.
    static constexpr float  BUG_CONTENT_CX = 905.f, BUG_CONTENT_CY = 611.f;
    static constexpr float  BUG_CONTENT_HX = 189.f, BUG_CONTENT_HY = 164.f;
    std::mt19937 bugRng{ std::random_device{}() };
    std::uniform_real_distribution<float> bugRoll{0.f, 1.f};
    // AUDIENCE_3 falls through to the exit on its own if left alone.
    double audience3EndTime = 0.0;
    static constexpr double AUDIENCE_3_HOLD_SEC = 15.0;

    // Zoom for stage n, counting down from the tightest shot: Z = k^(3-n).
    static float stageZoom(int n) {
        float z = 1.f;
        for (int i = n; i < 3; ++i) z /= CAM_ZOOM_STEP;
        return z;
    }
    // Live zoom, easing from the previous stage into the current one.
    float currentZoom() const {
        const float t  = stageT * stageT * (3.f - 2.f * stageT);   // smoothstep
        const float z0 = stageZoom(stage > 0 ? stage - 1 : 0);
        return z0 + (stageZoom(stage) - z0) * t;
    }

    static void mat3SetIdentity(float M[3][3]) {
        M[0][0]=1; M[0][1]=0; M[0][2]=0;
        M[1][0]=0; M[1][1]=1; M[1][2]=0;
        M[2][0]=0; M[2][1]=0; M[2][2]=1;
    }

    static void mat3Copy(float dst[3][3], const float src[3][3]) {
        memcpy(dst, src, sizeof(float) * 9);
    }

    // Translation matrix applied as a right-multiply on Hg_pixel — shifts the
    // texel that ends up at any given screen pixel by (dx, dy) in texel space.
    void mulHgByTranslation(float dx, float dy) {
        float Tcorr[3][3] = {{1,0,dx},{0,1,dy},{0,0,1}};
        float tmp[3][3];
        mat3Mul(Hg_pixel, Tcorr, tmp);
        mat3Copy(Hg_pixel, tmp);
    }

    // Pick the (texel-space) IDLE-frame centroid for the current direction
    // and step state, mirroring the IDLE rendering branch.
    void idleFrameTexel(bool isLeft, float& x, float& y) const {
        if (!firstStepTaken) {
            x = isLeft ? align::L_IDLE_FIRST_X : align::R_IDLE_FIRST_X;
            y = isLeft ? align::L_IDLE_FIRST_Y : align::R_IDLE_FIRST_Y;
        } else if ((stepCount % 2) == 0) {
            x = isLeft ? align::L_IDLE_EVEN_X : align::R_IDLE_EVEN_X;
            y = isLeft ? align::L_IDLE_EVEN_Y : align::R_IDLE_EVEN_Y;
        } else {
            x = isLeft ? align::L_IDLE_ODD_X : align::R_IDLE_ODD_X;
            y = isLeft ? align::L_IDLE_ODD_Y : align::R_IDLE_ODD_Y;
        }
    }

    // After a TURN finishes, the matrix on the IDLE frame is identical to the
    // matrix on the turn's last frame (Hg_turn was just folded). They will
    // visually jump unless the IDLE-frame texel matches the turn-last texel.
    // Apply a translation Δ = (turn_last_texel − idle_first_texel) so the
    // upcoming IDLE renders at the same screen pixel as the turn's final frame.
    void applyTurnIdleAlignment() {
        const bool isLeft = (appState == AppState::WALKING_LEFT);
        const float tx_turn = isLeft ? align::TURN_LEFT_LAST_X : align::TURN_RIGHT_LAST_X;
        const float ty_turn = isLeft ? align::TURN_LEFT_LAST_Y : align::TURN_RIGHT_LAST_Y;
        float tx_idle, ty_idle;
        idleFrameTexel(isLeft, tx_idle, ty_idle);
        const float dx = tx_turn - tx_idle, dy = ty_turn - ty_idle;
        mulHgByTranslation(dx, dy);
        std::cerr << "[align/post] isLeft=" << isLeft
                  << " stepsInCycle=" << stepsInCycle
                  << " stepCount=" << stepCount
                  << " Δ=(" << dx << "," << dy << ")"
                  << " Hg_pixel.Y=" << Hg_pixel[1][2]
                  << " Hg_lift.Y=" << Hg_lift[1][2] << "\n";
    }

    // Symmetric to the above but for the entry side: align the turn's FIRST
    // frame with the pre-turn IDLE frame so the transition INTO the turn is
    // also smooth. Called at the moment of direction switch, after Hg_turn
    // has been set and appState has been flipped (so we pass the OLD direction
    // as wasRight).
    void applyPreTurnAlignment(bool wasRight) {
        // Pre-turn IDLE texel: pull the OLD direction's idle centroid (the
        // frame we were displaying immediately before the turn started).
        float pre_x, pre_y;
        idleFrameTexel(/*isLeft=*/!wasRight, pre_x, pre_y);
        // Apply old direction's pre-transform if it was LEFT.
        if (!wasRight) {
            pre_x += hom::H_left_px_to_right[0][2];
            pre_y += hom::H_left_px_to_right[1][2];
        }

        // Turn first frame in NEW direction. After the switch, if we WERE
        // going right we're NOW going left and vice versa, so nowLeft = wasRight.
        const bool nowLeft = wasRight;
        float t_x = nowLeft ? align::TURN_LEFT_FIRST_X : align::TURN_RIGHT_FIRST_X;
        float t_y = nowLeft ? align::TURN_LEFT_FIRST_Y : align::TURN_RIGHT_FIRST_Y;
        if (nowLeft) {
            t_x += hom::H_left_px_to_right[0][2];
            t_y += hom::H_left_px_to_right[1][2];
        }

        // Apply Hg_turn (just-set partial-cycle correction) to the turn-first.
        const float vw = Hg_turn[2][0]*t_x + Hg_turn[2][1]*t_y + Hg_turn[2][2];
        const float vx = (Hg_turn[0][0]*t_x + Hg_turn[0][1]*t_y + Hg_turn[0][2]) / vw;
        const float vy = (Hg_turn[1][0]*t_x + Hg_turn[1][1]*t_y + Hg_turn[1][2]) / vw;

        // Δ = (T_pre_old · pre_idle) − (Hg_turn · T_pre_new · turn_first)
        const float dx = pre_x - vx, dy = pre_y - vy;
        mulHgByTranslation(dx, dy);
        std::cerr << "[align/pre] wasRight=" << wasRight
                  << " stepsInCycle=" << stepsInCycle
                  << " stepCount=" << stepCount
                  << " Δ=(" << dx << "," << dy << ")"
                  << " Hg_pixel.Y=" << Hg_pixel[1][2]
                  << " Hg_lift.Y=" << Hg_lift[1][2] << "\n";
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    void updateSpriteMatrix() {
        // Build the per-frame model matrix from the cumulative pixel-space
        // homography. During TURN, premultiply by Hg_turn (mid-pair half-step
        // correction). Left-walk sprites are then warped into the right-walk
        // frame via H_left_px_to_right so both directions share one frame.
        // Finally, Hg_lift is left-multiplied to apply the cumulative upward
        // shift in screen-pixel space.
        float Hp[3][3];
        if (walkSub == WalkSub::TURN) {
            mat3Mul(Hg_pixel, Hg_turn, Hp);
        } else {
            memcpy(Hp, Hg_pixel, sizeof(Hp));
        }
        // Walk_left frames are warped into walk_right space. CENTERING renders
        // the center_sprite in its native space (Hg_pixel was rebuilt to
        // absolute coordinates at the trigger).
        if (appState == AppState::WALKING_LEFT) {
            float Hp2[3][3];
            mat3Mul(Hp, hom::H_left_px_to_right, Hp2);
            memcpy(Hp, Hp2, sizeof(Hp));
        }
        float Hp_lifted[3][3];
        mat3Mul(Hg_lift, Hp, Hp_lifted);
        float Hn[3][3];
        homPixToNDC(Hp_lifted, Hn);
        spriteMatrix = embedHom(Hn);
    }

    // Used for hold-to-walk: polls directly rather than relying on callback flag
    bool rightHeld() const { return glfwGetKey(win,GLFW_KEY_RIGHT)==GLFW_PRESS; }
    bool leftHeld()  const { return glfwGetKey(win,GLFW_KEY_LEFT)==GLFW_PRESS; }

    void switchBgVideo(const std::string& path) {
        // Detach the mixer from bgVid while we tear it down and repopulate
        // audioBuf — the callback runs on a separate thread and would read
        // freed/partial data otherwise.
        VideoDecoder* prev = mixer.vidSrc;
        mixer.vidSrc = nullptr;
        bgVid.close();
        bgVid.eof=false; bgVid.loopVideo=false;
        bgVid.vsi=-1; bgVid.asi=-1; bgVid.w=0; bgVid.h=0;
        bgVid.audioPos.store(0);
        bgVid.audioBuf.clear();
        if (!bgVid.open(path,true)) {
            std::cerr<<"[bg] Cannot open "<<path<<" — falling back to grass.mp4\n";
            if (!bgVid.open("vids/splice/grass.mp4", true)) {
                std::cerr<<"[bg] Fallback also failed\n";
                mixer.vidSrc = prev;
                return;
            }
        }
        bgVid.loopVideo=true;
        bgVid.preDecodeAudio();
        if (bgVid.vsi>=0) {
            AVRational r=bgVid.fmt->streams[bgVid.vsi]->avg_frame_rate;
            if (r.num>0) bgVidFrameTime=(double)r.den/r.num;
        }
        // Reallocate texture + buffer if the new video's dimensions differ
        // from what was set up at init. Without this, switching to a
        // larger-resolution clip overflows the buffer and the texture upload
        // is rejected (leaving the previous clip's frame frozen on screen).
        const size_t needBytes = (size_t)bgVid.w * bgVid.h * 3;
        if (bgVidRgb.size() != needBytes) bgVidRgb.resize(needBytes);
        if ((int)bgVidTexW != bgVid.w || (int)bgVidTexH != bgVid.h) {
            if (bgVidTex) glDeleteTextures(1, &bgVidTex);
            bgVidTex  = makeTex(bgVid.w, bgVid.h);
            bgVidTexW = bgVid.w;
            bgVidTexH = bgVid.h;
        }
        bgVid.nextVideoFrame(bgVidRgb.data());
        glBindTexture(GL_TEXTURE_2D,bgVidTex);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,bgVid.w,bgVid.h,
                        GL_RGB,GL_UNSIGNED_BYTE,bgVidRgb.data());
        mixer.vidSrc = prev;
    }

    // From WALKING_LEFT IDLE: kick off the center_sprite transition. Hg_pixel
    // is rebuilt as a pure translation T_align so center_sprite[0] lands at the
    // same screen-pixel position as the walk_left idle frame; the per-frame
    // CENTERING update then linearly decays Hg_pixel and Hg_lift toward
    // identity over the 53-frame sequence so the final pose is canonical.
    void triggerCentering() {
        float ix, iy;
        idleFrameTexel(/*isLeft=*/true, ix, iy);
        // Walk_left frames render through H_left_px_to_right, which is a pure
        // translation in this codebase.
        float wx = ix + hom::H_left_px_to_right[0][2];
        float wy = iy + hom::H_left_px_to_right[1][2];
        // Apply Hg_pixel via the full 3x3 form (cheap and correctness-preserving).
        const float w = Hg_pixel[2][0]*wx + Hg_pixel[2][1]*wy + Hg_pixel[2][2];
        const float vx = (Hg_pixel[0][0]*wx + Hg_pixel[0][1]*wy + Hg_pixel[0][2]) / w;
        const float vy = (Hg_pixel[1][0]*wx + Hg_pixel[1][1]*wy + Hg_pixel[1][2]) / w;
        T_align_tx = vx - centerFirstX;
        T_align_ty = vy - centerFirstY;
        Hg_lift_start_y = Hg_lift[1][2];
        mat3SetIdentity(Hg_pixel);
        Hg_pixel[0][2] = T_align_tx;
        Hg_pixel[1][2] = T_align_ty;
        mat3SetIdentity(Hg_turn);
        seqCenter.reset();
        centeringFrame = 0;
        wavesAlpha = 0.5f;
        blackAlpha = 0.f;          // scene video fades to black across CENTERING
        walkClip.stop();
        // Centered portion plays the title-screen track (scene bg audio hands off).
        titleAudio.audioPos = 0;
        mixer.vidSrc  = &titleAudio;
        mixer.vidSrc2 = nullptr;
        mixer.bgGain.store(1.0f);
        walkSub  = WalkSub::IDLE;
        appState = AppState::CENTERING;
        anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
        pendingCenter = false;
    }

    // Drop every walk-cycle sprite sequence. Once the character has centred he
    // never walks again, and these are ~540 full-frame RGBA textures.
    void releaseWalkSequences() {
        seqBackPose.unload();  seqEnterRight.unload();
        seqLeftFoot.unload();  seqRightFoot.unload();  seqShakeHead.unload();
        seqTurnLeft.unload();  seqWLTurnRight.unload();
        seqWLLeftFoot.unload(); seqWLRightFoot.unload(); seqWLShakeHead.unload();
        // dirRight/dirLeft keep pointing at these now-empty sequences: current()
        // returns 0 and drawTex early-outs, which is safer than nulling them.
    }

    // Place an audience layer by the drawing's shared bottom-left corner, with
    // its bottom row on the screen bottom. All three levels are nested crops of
    // one drawing anchored at that corner — including the pre-split level-3
    // back layer, which is cropped to the level-2 rect and so shares it too —
    // which is why one expression positions every layer at any zoom.
    void drawAudience(BoilStill& still, float Z, float alpha, double now) {
        if (!still.loaded || alpha <= 0.f) return;
        const float as = Z * AUD_WORLD_SCALE;
        const float ax = CAM_FOCAL_X + Z * (AUD_WORLD_X - CAM_FOCAL_X);
        const float w  = as * still.w;
        const float h  = as * still.h;
        drawBoiledTinted(progBoilTint, quad, still.current(),
                         rectMatrix(ax, (float)cfg::WIN_H - h, w, h),
                         w, h, HAND_BOIL_PX,
                         (float)std::floor(now*HAND_BOIL_HZ), audTint, alpha);
    }

    // Screen rect the crowd fills at zoom Z, measured from the widest (level-3)
    // crop so it covers every level. Used to keep the bug inside the audience.
    void audienceRect(float Z, float& x0, float& y0, float& x1, float& y1) const {
        const float as = Z * AUD_WORLD_SCALE;
        const float w  = as * (stillAudL3Front.w ? stillAudL3Front.w : 1394);
        const float h  = as * (stillAudL3Front.h ? stillAudL3Front.h : 748);
        x0 = CAM_FOCAL_X + Z * (AUD_WORLD_X - CAM_FOCAL_X);
        x1 = x0 + w;
        y1 = (float)cfg::WIN_H;
        y0 = y1 - h;
    }

    // Hand off to the exit screen: the hand-drawn Exit over black, paced to
    // span the whole voice recording, with everything else released.
    void beginOutro(double now) {
        seqExit.reset();
        // The crowd and the bug are done; the man carries over — he stands
        // beside the sphere on the exit screen.
        stillAudL1.unload(); stillAudL2.unload();
        stillAudL3Back.unload(); stillAudL3Front.unload();
        seqSmoothBug.unload();
        seqBg.unload();          // waves overlay is deliberately absent here
        bgTex = 0;
        bugActive = false;

        exitAudio.audioPos = 0;
        mixer.vidSrc  = &exitAudio;
        mixer.vidSrc2 = nullptr;
        mixer.bgGain.store(EXIT_AUDIO_GAIN);
        // Run until the recording finishes, and spread the exit frames evenly
        // across it so the animation lasts the whole way rather than cutting to
        // black early.
        double dur = exitAudio.audioBuf.empty() ? 0.0
            : (double)exitAudio.audioBuf.size()
              / (double)(cfg::AUDIO_CH * cfg::AUDIO_RATE);
        if (dur <= 0.0) dur = (double)seqExit.frames.size() * OUTRO_FRAME_SEC;
        exitFrameSec = seqExit.frames.empty()
            ? OUTRO_FRAME_SEC : dur / (double)seqExit.frames.size();
        lastExitFrameTime = now;
        outroEndTime      = now + dur;
        appState          = AppState::OUTRO;
    }

    void sceneTransition() {
        // Scene transition instead of shake_head
        stepCount=0;
        stepsInCycle=0;
        mat3Copy(Hg_pixel, hom::H_right_base);
        mat3SetIdentity(Hg_turn);
        mat3SetIdentity(Hg_lift);
        spriteMatrix=glm::mat4(1.f);
        sceneIdx = (sceneIdx + 1) % (int)(sizeof(kScenes)/sizeof(kScenes[0]));
        const Scene& s = kScenes[sceneIdx];
        switchBgVideo(s.path);
        showWavesOverlay = s.showsWaves;
        walkSub=WalkSub::IDLE;
        walkClip.stop();
        return;
    }

    // ── Polymorphic walk primitives ──────────────────────────────────────────
    // Both WALKING_RIGHT and WALKING_LEFT share one flow; direction-specific
    // data (sprite sequences, step delta) lives in WalkDir, selected here.
    WalkDir& currentDir() {
        return appState==AppState::WALKING_RIGHT ? dirRight : dirLeft;
    }

    bool atWalkEnd() const {
        if (appState==AppState::WALKING_RIGHT) return stepCount > cfg::MAX_CYCLES*2;
        return stepCount <= 0;
    }

    void doWalkEnd() {
        if (appState==AppState::WALKING_RIGHT) { sceneTransition(); return; }
        currentDir().shakeHead->reset();
        walkSub = WalkSub::SHAKE_HEAD;
    }

    void startStep() {
        if (atWalkEnd()) { doWalkEnd(); return; }
        if (!firstStepTaken) {
            firstStepTaken=true;
            if (kScenes[sceneIdx].showsWaves) showWavesOverlay=true;
        }
        // Pair boundary: when the previous cycle has accumulated 2 foot
        // steps, advance Hg_pixel by the current direction's step matrix and
        // reset the cycle counter. Done here (start of next pair) so the IDLE
        // between pairs stays anchored at the just-finished position.
        if (stepsInCycle == 2) {
            const auto& step = (appState == AppState::WALKING_RIGHT)
                ? hom::H_right_step
                : hom::H_left_step;
            float tmp[3][3];
            mat3Mul(Hg_pixel, step, tmp);
            memcpy(Hg_pixel, tmp, sizeof(Hg_pixel));
            stepsInCycle = 0;
        }
        updateSpriteMatrix();
        WalkDir& d = currentDir();
        if ((stepCount%2)==0) { d.leftFoot->reset();  walkSub=WalkSub::LEFT_FOOT; }
        else                  { d.rightFoot->reset(); walkSub=WalkSub::RIGHT_FOOT; }
        walkClip.play(false);
    }

    void onStepComplete() {
        if (walkSub==WalkSub::SHAKE_HEAD || walkSub==WalkSub::TURN) {
            if (walkSub == WalkSub::TURN) {
                // Fold the mid-pair half-step correction into Hg_pixel so
                // subsequent steps in the new direction stack correctly,
                // then translate Hg_pixel so the upcoming IDLE frame lands
                // at the same screen pixel as the turn's last frame.
                float tmp[3][3];
                mat3Mul(Hg_pixel, Hg_turn, tmp);
                memcpy(Hg_pixel, tmp, sizeof(Hg_pixel));
                mat3SetIdentity(Hg_turn);
                applyTurnIdleAlignment();
            }
            walkSub = WalkSub::IDLE;
            updateSpriteMatrix();
            return;
        }
        stepCount += currentDir().stepDelta;
        stepsInCycle++;
        walkSub = WalkSub::IDLE;
        walkClip.stop();
    }

    // Advance the current sub-sequence one frame; return true if it finished.
    bool advSeq() {
        WalkDir& d = currentDir();
        SpriteSeq* s = nullptr;
        switch (walkSub) {
            case WalkSub::LEFT_FOOT:  s = d.leftFoot;  break;
            case WalkSub::RIGHT_FOOT: s = d.rightFoot; break;
            case WalkSub::SHAKE_HEAD: s = d.shakeHead; break;
            case WalkSub::TURN:       s = d.turnIn;    break;
            default: return false;
        }
        s->advance();
        return s->done;
    }

    // ── Key callback ─────────────────────────────────────────────────────────
    static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
        if (action!=GLFW_PRESS) return;
        auto* a=static_cast<App*>(glfwGetWindowUserPointer(w));
        if (!a) return;
        if (key==GLFW_KEY_ESCAPE){ glfwSetWindowShouldClose(w,GLFW_TRUE); return; }
        a->anyKeyPressed   = true;
        a->rightKeyPressed = (key==GLFW_KEY_RIGHT);
        a->leftKeyPressed  = (key==GLFW_KEY_LEFT);
        a->downKeyPressed  = (key==GLFW_KEY_DOWN);
    }

    // ── Upload helpers ────────────────────────────────────────────────────────
    void uploadIntroFrame() {
        if (!introVid.nextVideoFrame(introRgb.data())) return;
        glBindTexture(GL_TEXTURE_2D,introTex);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,introVid.w,introVid.h,
                        GL_RGB,GL_UNSIGNED_BYTE,introRgb.data());
    }

    void uploadBgVidFrame() {
        if (!bgVid.nextVideoFrame(bgVidRgb.data())) return; // loopVideo handles restart
        glBindTexture(GL_TEXTURE_2D,bgVidTex);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,bgVid.w,bgVid.h,
                        GL_RGB,GL_UNSIGNED_BYTE,bgVidRgb.data());
    }

    void tickBg() {
        if (seqBg.frames.empty()) return;
        if (bgCur>=(int)seqBg.frames.size()) bgCur=0;
        bgTex=seqBg.frames[bgCur++];
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

        prog        = makeProgram(VS_SRC,   FS_SRC);
        progOverlay = makeProgram(VS_SRC,   FS_OVERLAY);
        progText    = makeProgram(VS_SRC,   FS_TEXT);
        progPlain   = makeProgram(VS_PLAIN, FS_PLAIN);
        progBoil    = makeProgram(VS_SRC,   FS_BOIL);
        progBoilTint= makeProgram(VS_SRC,   FS_BOIL_TINT);
        progTint    = makeProgram(VS_SRC,   FS_TINT);
        quad.init();

        title.load(progText);

        // Title audio (looping background music)
        if (titleAudio.open("audios/trimmedsong.m4a",true))
            titleAudio.preDecodeAudio();
        else
            std::cerr<<"[audio] trimmedsong.m4a not found\n";

        // Exit voice recording — played during the OUTRO exit animation.
        if (exitAudio.open("audios/exit_recording.m4a",true))
            exitAudio.preDecodeAudio();
        else
            std::cerr<<"[audio] exit_recording.m4a not found\n";

        if (!introVid.open("vids/splice/new_intro.mp4",true)) return false;
        introTex=makeTex(introVid.w,introVid.h);
        introRgb.resize((size_t)introVid.w*introVid.h*3);
        if (introVid.vsi>=0) {
            AVRational r=introVid.fmt->streams[introVid.vsi]->avg_frame_rate;
            if (r.num>0) introFrameTime=(double)r.den/r.num;
        }
        // Deferred: introVid.preDecodeAudio() costs ~1 sec for a 108-sec clip
        // and the intro doesn't play until the user dismisses the title.
        // We pre-decode it lazily on the TITLE → INTRO transition.

        // Background video (station.mp4, looping) — audio mixed in under walk SFX
        if (!bgVid.open("vids/splice/station.mp4",true))
            std::cerr<<"[bg] Cannot open station.mp4\n";
        bgVid.loopVideo=true;
        bgVid.preDecodeAudio();
        if (bgVid.w>0 && bgVid.h>0) {
            bgVidTex=makeTex(bgVid.w,bgVid.h);
            bgVidTexW=bgVid.w; bgVidTexH=bgVid.h;
            bgVidRgb.resize((size_t)bgVid.w*bgVid.h*3);
            if (bgVid.vsi>=0) {
                AVRational r=bgVid.fmt->streams[bgVid.vsi]->avg_frame_rate;
                if (r.num>0) bgVidFrameTime=(double)r.den/r.num;
            }
            // Prime first frame
            bgVid.nextVideoFrame(bgVidRgb.data());
            glBindTexture(GL_TEXTURE_2D,bgVidTex);
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,bgVid.w,bgVid.h,
                            GL_RGB,GL_UNSIGNED_BYTE,bgVidRgb.data());
        }

        // Background waves (now used as overlay, not primary bg)
        if (!seqBg.load("renders/background_waves_staggered"))
            std::cerr<<"[bg] No frames in renders/background_waves_staggered\n";

        // Title-screen ant frames, looped beneath the terminal text. Generated
        // by ants_smooth.py: the isolation pass's wavy top edge, smoothed and
        // feathered, with the original footage filling everything below it.
        if (!seqAnts.load("renders/ants_smooth"))
            std::cerr<<"[title] No ant frames in renders/ants_smooth (run ants_smooth.py)\n";

        seqBackPose.load("renders/back_pose");
        seqEnterRight.load("renders/enter_right");
        seqLeftFoot.load("renders/walk_right/left_foot");
        seqRightFoot.load("renders/walk_right/right_foot");
        seqShakeHead.load("renders/walk_right/shake_head");
        seqTurnLeft.load("renders/walk_left/turn_left");

        seqWLTurnRight.load("renders/walk_right/turn_right");
        seqWLLeftFoot.load("renders/walk_left/left_foot");
        seqWLRightFoot.load("renders/walk_left/right_foot");
        seqWLShakeHead.load("renders/walk_left/shake_head");

        dirRight = { &seqLeftFoot,   &seqRightFoot,   &seqShakeHead,   &seqWLTurnRight, +1 };
        dirLeft  = { &seqWLLeftFoot, &seqWLRightFoot, &seqWLShakeHead, &seqTurnLeft,    -1 };

        // Center transition + idle pose
        if (!seqCenter.load("renders/center_sprite"))
            std::cerr<<"[center] No frames in renders/center_sprite\n";
        if (!computeAlphaCentroidBottom("renders/center_sprite/0000.png",
                                        centerFirstX, centerFirstY)) {
            std::cerr<<"[center] Failed to read center_sprite/0000.png centroid\n";
            // Fallback: assume canonical screen center.
            centerFirstX = (float)cfg::WIN_W * 0.5f;
            centerFirstY = (float)cfg::WIN_H * 0.5f;
        }
        // Find and measure the LAST center_sprite frame — that's what the
        // centered pose renders, and its character extends much further
        // down in the texture than the first frame, so the end-state matrix
        // must be anchored on it (not on frame 0) to keep the scaled feet
        // flush with the bottom of the screen.
        {
            std::string lastPath;
            for (auto& e : fs::directory_iterator("renders/center_sprite")) {
                auto fn = e.path().filename().string();
                if (e.path().extension() == ".png" && fn.substr(0,2) != "._") {
                    if (lastPath.empty() ||
                        fn > fs::path(lastPath).filename().string())
                        lastPath = e.path().string();
                }
            }
            if (lastPath.empty() ||
                !computeAlphaCentroidBottom(lastPath, centerLastX, centerLastY)) {
                std::cerr<<"[center] Failed to read last center_sprite centroid\n";
                centerLastX = centerFirstX;
                centerLastY = centerFirstY;
            }
        }

        // Hand-drawn pull-back stills, boiled at draw time. The man's anchor is
        // measured the same way the walk sequences are — alpha centroid X +
        // bbox-bottom Y — so he can be planted by his feet at any zoom, and it
        // is measured on the still that actually gets drawn.
        if (!stillMan.load("renders/stills/man.png"))
            std::cerr<<"[hand] No renders/stills/man.png (run boil_stills.py)\n";
        if (!computeAlphaCentroidBottom("renders/stills/man.png",
                                        manCentroidX, manBottomY)) {
            std::cerr<<"[hand] Failed to read stills/man.png centroid\n";
            manCentroidX = stillMan.w * 0.5f;
            manBottomY   = (float)stillMan.h;
        }
        if (!stillAudL1.load("renders/stills/aud_l1.png"))
            std::cerr<<"[aud] No renders/stills/aud_l1.png (run boil_stills.py)\n";
        if (!stillAudL2.load("renders/stills/aud_l2.png"))
            std::cerr<<"[aud] No renders/stills/aud_l2.png (run boil_stills.py)\n";
        if (!stillAudL3Back.load("renders/stills/aud_l3_back.png"))
            std::cerr<<"[aud] No renders/stills/aud_l3_back.png (run boil_stills.py)\n";
        if (!stillAudL3Front.load("renders/stills/aud_l3_front.png"))
            std::cerr<<"[aud] No renders/stills/aud_l3_front.png (run boil_stills.py)\n";

        // Bug clip frames (smooth_bug PNG sequence) — played during AUDIENCE_3.
        if (!seqSmoothBug.load("vids/moving_objects/bug/smooth_bug"))
            std::cerr<<"[bug] No frames in vids/moving_objects/bug/smooth_bug\n";

        // Exit animation frames (exit/ PNG sequence) — the final outro screen.
        if (!seqExit.load("exit"))
            std::cerr<<"[exit] No frames in exit/\n";

        // Exit-screen stills. The man's bbox lets him stand on a ground line
        // rather than hang off his centroid; the exit animation's bbox is taken
        // from its LAST frame, the only one with the drawing fully in.
        texExitDisc  = loadTexture("renders/exit_screen/disc.png");
        texTreeClean = loadTexture("renders/exit_screen/tree_clean.png");
        texTreeNoise = loadTexture("renders/exit_screen/tree_noise.png");
        texRim       = loadTexture("renders/exit_screen/rim.png");
        if (!texExitDisc || !texTreeClean || !texRim)
            std::cerr<<"[exit] Missing renders/exit_screen (run exit_screen_assets.py)\n";
        if (!computeAlphaBBox("renders/stills/man.png",
                              manFigX0, manFigY0, manFigX1, manFigY1)) {
            std::cerr<<"[exit] Failed to measure stills/man.png bbox\n";
            manFigX0 = 0.f; manFigY0 = 0.f;
            manFigX1 = (float)stillMan.w; manFigY1 = (float)stillMan.h;
        }
        {
            const std::string lastExit = lastPngIn("exit");
            if (lastExit.empty() ||
                !computeAlphaBBox(lastExit, exitBoxX0, exitBoxY0, exitBoxX1, exitBoxY1)) {
                std::cerr<<"[exit] Failed to measure exit/ bbox\n";
                exitBoxX0 = 0.f; exitBoxY0 = 0.f;
                exitBoxX1 = (float)cfg::WIN_W; exitBoxY1 = (float)cfg::WIN_H;
            }
        }

        if (!walkClip.load("audios/walking.m4a"))
            std::cerr<<"[audio] walking.m4a not found\n";

        mixer.vidSrc   = &titleAudio;    // looping title music
        mixer.walkPcm  = &walkClip.pcm;
        mixer.walkPos  = &walkClip.pos;
        mixer.walkPlay = &walkClip.playing;
        mixer.walkLoop = &walkClip.looping;
        if (!mixer.init()) std::cerr<<"[audio] mixer init failed\n";

        double t0=glfwGetTime();
        lastIntroTime=lastBgTime=lastBgVidTime=lastSpriteTime=lastAntTime=t0;
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
            bool advAnt = (now-lastAntTime)    >= antFrameTime;
            bool advBgV = (now-lastBgVidTime)  >= bgVidFrameTime;
            bool advS   = (now-lastSpriteTime) >= spriteFrameTime;
            if (advI)   lastIntroTime  = now;
            if (advBg)  { lastBgTime   = now; tickBg(); }
            if (advBgV) { lastBgVidTime = now; if (kScenes[sceneIdx].animatesBg) uploadBgVidFrame(); }
            if (advS)   lastSpriteTime = now;
            // Advance ant frames only while the title is on screen; loop the
            // sequence when it reaches the end.
            if (advAnt && appState == AppState::TITLE && seqAnts.loaded) {
                lastAntTime = now;
                if (seqAnts.done) seqAnts.reset();
                seqAnts.advance();
            }

            // ── State machine ─────────────────────────────────────────────────
            switch (appState) {

            case AppState::TITLE:
                title.update(now);
                // Only accept key after typing finishes (prevents accidental skip)
                if (anyKeyPressed && title.phase==TitleScreen::Phase::ELLIPSIS) {
                    anyKeyPressed=false; rightKeyPressed=false; leftKeyPressed=false;
                    // Deferred audio pre-decode — done once on first transition.
                    if (introVid.audioBuf.empty()) introVid.preDecodeAudio();
                    introVid.audioPos=0;
                    mixer.vidSrc=&introVid;   // start intro audio
                    uploadIntroFrame();        // prime first frame — no black flash
                    lastIntroTime=now;
                    seqAnts.unload();          // title-only, ~120 full-frame RGBA
                    appState=AppState::INTRO;
                }
                break;

            case AppState::INTRO:
                if (advI) uploadIntroFrame();
                if (introVid.eof || anyKeyPressed) {
                    anyKeyPressed=false; rightKeyPressed=false; leftKeyPressed=false;
                    // Intro never replays: release its texture, RGB staging
                    // buffer and the fully-decoded 108-sec audio track. Detach
                    // the mixer first and reattach after — its callback runs on
                    // another thread and would read freed data otherwise (same
                    // ordering switchBgVideo() uses).
                    mixer.vidSrc=nullptr;
                    introVid.close();
                    introVid.audioBuf.clear();
                    introVid.audioBuf.shrink_to_fit();
                    introRgb.clear(); introRgb.shrink_to_fit();
                    if (introTex) { glDeleteTextures(1,&introTex); introTex=0; }
                    bgVid.audioPos=0;         // restart bg audio from top
                    mixer.vidSrc=&bgVid;      // bg video audio takes over
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
                    seqEnterRight.advance();
                    if (seqEnterRight.done) {
                            stepCount=0; updateSpriteMatrix();
                            walkSub=WalkSub::IDLE; appState=AppState::WALKING_RIGHT;
                        }
                    }
                break;

            case AppState::WALKING_RIGHT:
            case AppState::WALKING_LEFT: {
                const bool isRight     = (appState==AppState::WALKING_RIGHT);
                // DOWN ("back") in walk_left → defer to next IDLE then enter CENTERING.
                // Always clear so a stale DOWN from walk_right can't bleed through.
                if (downKeyPressed) {
                    if (!isRight) pendingCenter = true;
                    downKeyPressed = false;
                }
                const bool sameKey     = isRight ? rightKeyPressed : leftKeyPressed;
                const bool oppositeKey = isRight ? leftKeyPressed  : rightKeyPressed;

                if (advS && walkSub != WalkSub::IDLE) {
                    if (advSeq()) {
                        onStepComplete();                      // apply stepDelta, → IDLE
                        // Auto-fire scene transition after step (MAX_CYCLES*2 + 1)
                        // — i.e., 13 steps — without waiting for another keypress.
                        if (isRight && atWalkEnd()) { doWalkEnd(); break; }
                        if (pendingCenter && !isRight) { triggerCentering(); break; }
                        if (oppositeKey) {
                            rightKeyPressed = leftKeyPressed = false;
                            // Mid-pair turn (one foot stepped in current cycle):
                            // capture half-step correction for the OUTGOING
                            // direction so TURN renders at the actual cycle
                            // position. Pair-boundary turns need no correction.
                            if (stepsInCycle == 1) {
                                const auto& halfStep = isRight
                                    ? hom::H_right_step_half
                                    : hom::H_left_step_half;
                                memcpy(Hg_turn, halfStep, sizeof(Hg_turn));
                            } else {
                                mat3SetIdentity(Hg_turn);
                            }
                            appState = isRight ? AppState::WALKING_LEFT
                                               : AppState::WALKING_RIGHT;
                            walkSub  = WalkSub::TURN;
                            currentDir().turnIn->reset();
                            applyPreTurnAlignment(isRight);
                            updateSpriteMatrix();   // direction-dependent pre-transform
                        } else if (sameKey) {
                            rightKeyPressed = leftKeyPressed = false;
                            startStep();
                        }
                    }
                }

                if (walkSub == WalkSub::IDLE) {
                    if (pendingCenter && !isRight) { triggerCentering(); break; }
                    if (sameKey) {
                        rightKeyPressed = leftKeyPressed = false;
                        startStep();
                    } else if (oppositeKey) {
                        rightKeyPressed = leftKeyPressed = false;
                        // Same mid-pair vs pair-boundary correction as above.
                        if (stepsInCycle == 1) {
                            const auto& halfStep = isRight
                                ? hom::H_right_step_half
                                : hom::H_left_step_half;
                            memcpy(Hg_turn, halfStep, sizeof(Hg_turn));
                        } else {
                            mat3SetIdentity(Hg_turn);
                        }
                        appState = isRight ? AppState::WALKING_LEFT
                                           : AppState::WALKING_RIGHT;
                        walkSub  = WalkSub::TURN;
                        currentDir().turnIn->reset();
                        applyPreTurnAlignment(isRight);
                        updateSpriteMatrix();   // direction-dependent pre-transform
                    }
                }
                break;
            }

            case AppState::CENTERING: {
                if (advS) {
                    seqCenter.advance();
                    centeringFrame++;
                    int N = (int)seqCenter.frames.size();
                    float t = (N <= 1) ? 1.f : (float)centeringFrame / (float)(N - 1);
                    if (t > 1.f) t = 1.f;
                    // Lerp from initial (scale 1, T_align translation) to a
                    // scaled, centered, bottom-anchored final state. The end
                    // matrix sends the LAST center_sprite frame's centroid X
                    // and bbox-bottom Y — i.e. where the character actually
                    // stands at the end of the animation — to (screen center
                    // X, screen bottom Y), scaled by CENTER_END_SCALE around
                    // that anchor so the feet stay at the bottom of the screen.
                    static constexpr float CENTER_END_SCALE = 0.75f;
                    const float s = 1.f - t * (1.f - CENTER_END_SCALE);
                    const float endTx = (float)cfg::WIN_W * 0.5f - CENTER_END_SCALE * centerLastX;
                    const float endTy = (float)cfg::WIN_H        - CENTER_END_SCALE * centerLastY;
                    Hg_pixel[0][0] = s;
                    Hg_pixel[0][1] = 0.f;
                    Hg_pixel[0][2] = (1.f - t) * T_align_tx + t * endTx;
                    Hg_pixel[1][0] = 0.f;
                    Hg_pixel[1][1] = s;
                    Hg_pixel[1][2] = (1.f - t) * T_align_ty + t * endTy;
                    Hg_pixel[2][0] = 0.f;
                    Hg_pixel[2][1] = 0.f;
                    Hg_pixel[2][2] = 1.f;
                    Hg_lift[1][2]  = (1.f - t) * Hg_lift_start_y;
                    wavesAlpha = 0.5f + 1.5f * t;
                    blackAlpha = t;                 // scene video fades to black
                    // Backdrop pushes in as the character settles.
                    wavesScale = 1.f + t * (WAVES_CENTER_ZOOM - 1.f);
                    if (seqCenter.done) {
                        // Lock the final scaled, foot-on-screen-bottom matrix.
                        Hg_pixel[0][0] = CENTER_END_SCALE;
                        Hg_pixel[0][1] = 0.f;
                        Hg_pixel[0][2] = endTx;
                        Hg_pixel[1][0] = 0.f;
                        Hg_pixel[1][1] = CENTER_END_SCALE;
                        Hg_pixel[1][2] = endTy;
                        Hg_pixel[2][0] = 0.f;
                        Hg_pixel[2][1] = 0.f;
                        Hg_pixel[2][2] = 1.f;
                        mat3SetIdentity(Hg_lift);
                        wavesAlpha = 2.0f;
                        blackAlpha = 1.0f;          // background fully black now
                        wavesScale = WAVES_CENTER_ZOOM;
                        // Hold on the centered pose with the wave overlay still
                        // running; wait for a key press before the exit. Clear
                        // any keys pressed mid-turn so the hold isn't skipped.
                        anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
                        appState = AppState::CENTER_HOLD;
                    }
                }
                break;
            }

            case AppState::CENTER_HOLD: {
                // Centered pose held over black while the waves keep looping.
                // DOWN swaps the digital sprite for the hand-drawn one — same
                // spot, standing taller — and begins the pull-back.
                if (downKeyPressed) {
                    anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
                    // The digital walk/centre sprites are behind us for good.
                    releaseWalkSequences();
                    seqCenter.unload();
                    stage = 0;
                    stageT = 1.f;              // straight swap, no zoom
                    stageStartTime = now;
                    // Hard cut, not a fade: the scene video comes straight back
                    // from under the black and the green wash lands with it, in
                    // the same frame the sprite becomes hand-drawn.
                    blackAlpha = 0.f;
                    greenAlpha = GREEN_MAX;
                    appState = AppState::HAND_DRAWN;
                }
                break;
            }

            case AppState::HAND_DRAWN:
            case AppState::AUDIENCE_1:
            case AppState::AUDIENCE_2:
            case AppState::AUDIENCE_3: {
                // Ease the camera into the current stage.
                const bool zooming = (stageT < 1.f);
                if (zooming) {
                    stageT = (float)((now - stageStartTime) / STAGE_XFADE_SEC);
                    if (stageT >= 1.f) {
                        stageT = 1.f;
                        // The level we just faded out of will never be shown again.
                        if (appState == AppState::AUDIENCE_2)      stillAudL1.unload();
                        else if (appState == AppState::AUDIENCE_3) stillAudL2.unload();
                        if (appState == AppState::AUDIENCE_3)
                            audience3EndTime = now + AUDIENCE_3_HOLD_SEC;
                        anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
                    }
                }

                // Backdrop pulls back with the camera, in proportion — the
                // waves shrink by exactly the factor the world does. Updated
                // mid-zoom too, so the backdrop eases with everything else.
                wavesScale = WAVES_CENTER_ZOOM * (currentZoom() / stageZoom(0));

                // Keys are ignored until the zoom settles so a fast
                // double-press can't skip one.
                if (zooming) break;

                // The hand-drawn art has no frames to advance any more — its
                // boil is driven straight off `now`, quantised to HAND_BOIL_HZ.

                if (appState == AppState::AUDIENCE_3) {
                    // Widest shot: a bug wanders through at random. Roll once a
                    // second, but only while the previous one has finished, so
                    // there is never more than one on screen.
                    if (!bugActive && (now - lastBugRoll) >= BUG_ROLL_SEC) {
                        lastBugRoll = now;
                        if (!seqSmoothBug.frames.empty() && bugRoll(bugRng) < BUG_PROB) {
                            seqSmoothBug.reset();
                            bugActive = true;
                            lastOutroFrameTime = now;
                            // Drop it somewhere in the crowd, inset far enough
                            // that the clip's whole travelled area stays inside.
                            float x0,y0,x1,y1;
                            audienceRect(stageZoom(3), x0,y0,x1,y1);
                            x0 += BUG_SCALE*BUG_CONTENT_HX;
                            x1 -= BUG_SCALE*BUG_CONTENT_HX;
                            y0 += BUG_SCALE*BUG_CONTENT_HY;
                            y1 -= BUG_SCALE*BUG_CONTENT_HY;
                            bugPosX = x0 + bugRoll(bugRng)*std::max(0.f, x1-x0);
                            bugPosY = y0 + bugRoll(bugRng)*std::max(0.f, y1-y0);
                        }
                    }
                    if (bugActive && (now - lastOutroFrameTime) >= OUTRO_FRAME_SEC) {
                        lastOutroFrameTime = now;
                        seqSmoothBug.advance();
                        if (seqSmoothBug.done) bugActive = false;
                    }
                    // Any key — or just waiting long enough — heads for the exit.
                    if (anyKeyPressed || now >= audience3EndTime) {
                        anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
                        beginOutro(now);
                    }
                } else if (downKeyPressed) {
                    anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
                    ++stage;
                    stageT = 0.f;
                    stageStartTime = now;
                    appState = (stage == 1) ? AppState::AUDIENCE_1
                             : (stage == 2) ? AppState::AUDIENCE_2
                                            : AppState::AUDIENCE_3;
                }
                break;
            }

            case AppState::OUTRO: {
                // The hand-drawn Exit over black, timed to span the whole voice
                // recording. It cuts to black once it has played through; the
                // recording keeps going, and the app quits when it ends.
                if ((now - lastExitFrameTime) >= exitFrameSec) {
                    lastExitFrameTime = now;
                    seqExit.advance();
                }
                if (now >= outroEndTime)
                    glfwSetWindowShouldClose(win, GLFW_TRUE);
                break;
            }
            }

            // Per-sprite-frame vertical lift, rate matched to the current
            // animation's natural bbox-bottom descent. Excluded from
            // ENTERING_RIGHT (the entry asset's drop is the intended walk-on
            // motion, not drift) and from IDLE (otherwise idling drifts up).
            if (advS && walkSub != WalkSub::IDLE &&
                (appState == AppState::WALKING_RIGHT ||
                 appState == AppState::WALKING_LEFT)) {
                float rate;
                if (walkSub == WalkSub::TURN) {
                    rate = (appState == AppState::WALKING_LEFT)
                             ? LIFT_TURN_LEFT : LIFT_TURN_RIGHT;
                } else {
                    rate = (appState == AppState::WALKING_LEFT)
                             ? LIFT_WALK_LEFT : LIFT_WALK_RIGHT;
                }
                Hg_lift[1][2] -= rate;
                // Send turn_left back: counter the −17.52 px X drift across the
                // turn so the final turn frame's center returns to the pre-turn
                // idle center.
                if (walkSub == WalkSub::TURN && appState == AppState::WALKING_LEFT) {
                    Hg_lift[0][2] += LIFT_TURN_LEFT_X;
                }
            }
            // Re-derive spriteMatrix every render frame so any matrix change
            // (Hg_pixel multiply, Hg_turn fold, alignment translations,
            // Hg_lift accumulation) is reflected before drawing.
            updateSpriteMatrix();

            // ── Render ───────────────────────────────────────────────────────
            if (appState==AppState::TITLE) {
                // Black terminal: ant frames are 1920×1080 but the ant content
                // only spans Y=0..839, so shift the texture down 241 px to
                // anchor the ants at the bottom of the screen.
                glClearColor(0.f, 0.f, 0.f, 1.f);
                glClear(GL_COLOR_BUFFER_BIT);
                glDisable(GL_BLEND);
                if (seqAnts.loaded) {
                    // Shift down so the smoothed horizon sits low in the frame,
                    // leaving the terminal text room above it. The footage now
                    // runs to the bottom of the texture, so nothing black comes
                    // up from underneath.
                    static constexpr float ANT_SHIFT_Y = 345.f;
                    float Hp[3][3] = {{1,0,0},{0,1,ANT_SHIFT_Y},{0,0,1}};
                    float Hn[3][3]; homPixToNDC(Hp, Hn);
                    drawTex(prog, quad, seqAnts.current(), embedHom(Hn));
                }
                glEnable(GL_BLEND);
                title.render();
            } else if (appState==AppState::OUTRO) {
                // Tree sphere + man, centred and 3/4 of the screen tall, over
                // plain black, with the exit animation playing across the
                // sphere. The animation stops drawing the instant it has played
                // through once — no freeze/hold — while the exit recording keeps
                // playing underneath.
                glClearColor(0,0,0,1);
                glClear(GL_COLOR_BUFFER_BIT);
                glEnable(GL_BLEND);

                // Vertical stack: exit text, gap, then the sphere.
                const float textH = EXIT_ANIM_FRAC * EXIT_STACK_H;
                const float vgap  = EXIT_VGAP_FRAC * EXIT_STACK_H;
                const float D     = EXIT_STACK_H - textH - vgap;   // sphere Ø
                const float figH  = std::max(1.f, manFigY1 - manFigY0);
                const float figW  = std::max(1.f, manFigX1 - manFigX0);
                const float ms    = (EXIT_MAN_FRAC * D) / figH;    // man tex scale
                const float manW  = ms * figW;
                const float gap   = EXIT_GAP_FRAC * D;
                // Centre the sphere+man block, and the stack as a whole.
                const float blockW = D + gap + manW;
                const float bx = ((float)cfg::WIN_W - blockW) * 0.5f;
                const float sy = ((float)cfg::WIN_H - EXIT_STACK_H) * 0.5f
                               + textH + vgap;                     // sphere top

                // Sphere: white field, the tree dark on it, its noise twin
                // boiled over that, then the drawn rim.
                drawTinted(progTint,quad,texExitDisc,rectMatrix(bx,sy,D,D),audTint);
                drawTinted(progTint,quad,texTreeClean,rectMatrix(bx,sy,D,D),exitInkTint);
                const int st = (int)std::fmod(now / EXIT_BOIL_SEC, 4.0);
                drawTinted(progTint,quad,texTreeNoise,
                           rectMatrix(bx + kBoilDX[st]*EXIT_BOIL_OFF,
                                      sy + kBoilDY[st]*EXIT_BOIL_OFF, D, D),
                           exitInkTint, kBoilAlpha[st]);
                drawTinted(progTint,quad,texRim,rectMatrix(bx,sy,D,D),exitInkTint);

                // Man to the right, on the black, standing on the sphere's
                // bottom edge — white so he reads against it.
                if (stillMan.loaded) {
                    const float mw = ms*stillMan.w, mh = ms*stillMan.h;
                    drawBoiledTinted(progBoilTint,quad,stillMan.current(),
                                     rectMatrix(bx + D + gap - ms*manFigX0,
                                                sy + D      - ms*manFigY1, mw, mh),
                                     mw, mh, HAND_BOIL_PX,
                                     (float)std::floor(now*HAND_BOIL_HZ), audTint);
                }

                // Exit animation above the whole block, centred on it and sized
                // off its drawn extent so the scaling doesn't depend on the
                // frame's empty margins.
                if (!seqExit.done) {
                    if (GLuint exTx = seqExit.current()) {
                        const float es = textH / std::max(1.f, exitBoxY1 - exitBoxY0);
                        drawTex(prog, quad, exTx,
                                rectMatrix(bx + blockW*0.5f - es*(exitBoxX0+exitBoxX1)*0.5f,
                                           sy - vgap - textH - es*exitBoxY0,
                                           es*(float)cfg::WIN_W, es*(float)cfg::WIN_H));
                    }
                }
            } else {
                glClearColor(0,0,0,1);
                glClear(GL_COLOR_BUFFER_BIT);
                // ── Background layer ─────────────────────────────────────────
                glDisable(GL_BLEND);
                if (appState==AppState::INTRO) {
                    drawTex(prog,quad,introTex);
                } else {
                    // Primary background: looping video (station or grass)
                    drawTex(prog,quad,bgVidTex);
                    glEnable(GL_BLEND);

                    // Fade the scene video to black. `blackAlpha` ramps 0→1
                    // across CENTERING so the video disappears under the waves
                    // while the character turns to face front. Drawn over the
                    // video but under the waves overlay so the waves "remain".
                    if (blackAlpha > 0.f) {
                        glUseProgram(progPlain);
                        glUniformMatrix4fv(glGetUniformLocation(progPlain,"uModel"),
                                           1,GL_FALSE,glm::value_ptr(glm::mat4(1.f)));
                        glm::vec4 black(0.f,0.f,0.f,blackAlpha);
                        glUniform4fv(glGetUniformLocation(progPlain,"uColor"),
                                     1,glm::value_ptr(black));
                        quad.draw();
                    }

                    // Waves overlay — `wavesAlpha` ramps 0.5 (walking) → 2.0
                    // (CENTERING end). progOverlay's threshold-amplify shader
                    // saturates wave regions to fully opaque at 2.0 while
                    // keeping the PNG's transparent regions transparent. Shown
                    // during the centering turn and the CENTER_HOLD wait, but
                    // NOT during the exit (the OUTRO) — that is clean black.
                    bool wavesActive = showWavesOverlay
                        || appState==AppState::CENTERING
                        || appState==AppState::CENTER_HOLD
                        || appState==AppState::HAND_DRAWN
                        || appState==AppState::AUDIENCE_1
                        || appState==AppState::AUDIENCE_2
                        || appState==AppState::AUDIENCE_3;
                    // Scaled about the screen centre: the backdrop pushes in
                    // over CENTERING and pulls back with each zoom-out, in step
                    // with the camera. wavesScale never drops below 1, so the
                    // texture always covers the frame.
                    if (wavesActive && bgTex) {
                        const float ws = wavesScale;
                        const glm::mat4 wm = (ws == 1.f) ? glm::mat4(1.f)
                            : rectMatrix((float)cfg::WIN_W * 0.5f * (1.f - ws),
                                         (float)cfg::WIN_H * 0.5f * (1.f - ws),
                                         (float)cfg::WIN_W * ws,
                                         (float)cfg::WIN_H * ws);
                        drawTex(progOverlay,quad,bgTex,wm,wavesAlpha);
                    }
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
                        drawTex(prog,quad,seqBackPose.current(),spriteMatrix);
                    else
                        drawPlaceholder({0.2f,0.2f,0.8f,0.4f},spriteMatrix);
                    break;

                case AppState::ENTERING_RIGHT:
                    if (seqEnterRight.loaded)
                        drawTex(prog,quad,seqEnterRight.current(),spriteMatrix);
                    else
                        drawPlaceholder({0.2f,0.8f,0.2f,0.4f},spriteMatrix);
                    break;

                // The digital sprite picks up a hand-drawn line wobble as it
                // turns to face front and holds, foreshadowing the swap to the
                // actual hand-drawn sprite. Amplitude is in screen pixels, so
                // it is divided by the sprite matrix's live X scale.
                case AppState::CENTERING:
                    if (seqCenter.loaded)
                        drawBoiled(progBoil,quad,seqCenter.current(),spriteMatrix,
                                   BOIL_PX,Hg_pixel[0][0],
                                   (float)std::floor(now*BOIL_HZ));
                    break;

                case AppState::CENTER_HOLD:
                    if (seqCenter.loaded)
                        drawBoiled(progBoil,quad,seqCenter.lastFrame(),spriteMatrix,
                                   BOIL_PX,Hg_pixel[0][0],
                                   (float)std::floor(now*BOIL_HZ));
                    break;

                // ── Hand-drawn pull-back ─────────────────────────────────────
                // One camera zoom Z drives the whole shot. The audience levels
                // are nested crops sharing a bottom-left world origin, so a
                // cross-fade between two of them moves nothing that both draw —
                // only the ink of the newly revealed rows fades up.
                case AppState::HAND_DRAWN:
                case AppState::AUDIENCE_1:
                case AppState::AUDIENCE_2:
                case AppState::AUDIENCE_3: {
                    const float Z = currentZoom();
                    const float t = stageT*stageT*(3.f-2.f*stageT);   // smoothstep

                    // Outgoing level fades out, incoming fades in. Level 3 is
                    // split so the bug can sit between its near and far ink.
                    if (appState==AppState::AUDIENCE_2 || appState==AppState::AUDIENCE_3)
                        drawAudience(appState==AppState::AUDIENCE_2 ? stillAudL1 : stillAudL2,
                                     Z, 1.f - t, now);
                    if (appState==AppState::AUDIENCE_1)
                        drawAudience(stillAudL1, Z, t, now);
                    else if (appState==AppState::AUDIENCE_2)
                        drawAudience(stillAudL2, Z, t, now);
                    else if (appState==AppState::AUDIENCE_3) {
                        drawAudience(stillAudL3Back, Z, t, now);
                        if (bugActive)
                            if (GLuint bugTx = seqSmoothBug.current())
                                drawTex(prog, quad, bugTx,
                                        rectMatrix(bugPosX - BUG_SCALE*BUG_CONTENT_CX,
                                                   bugPosY - BUG_SCALE*BUG_CONTENT_CY,
                                                   BUG_SCALE*(float)cfg::WIN_W,
                                                   BUG_SCALE*(float)cfg::WIN_H),
                                        BUG_ALPHA);
                        drawAudience(stillAudL3Front, Z, t, now);
                    }

                    // The man draws last: he is never occluded by the crowd. He
                    // is his own black ink here, not recoloured like the crowd,
                    // so the tint just restates what the still already carries.
                    if (stillMan.loaded) {
                        const float ms = Z * MAN_WORLD_SCALE;
                        const float mx = CAM_FOCAL_X + Z * (MAN_WORLD_X - CAM_FOCAL_X);
                        const float mw = ms*stillMan.w, mh = ms*stillMan.h;
                        drawBoiledTinted(progBoilTint, quad, stillMan.current(),
                                         rectMatrix(mx - ms*manCentroidX,
                                                    (float)cfg::WIN_H - ms*manBottomY,
                                                    mw, mh),
                                         mw, mh, HAND_BOIL_PX,
                                         (float)std::floor(now*HAND_BOIL_HZ),
                                         glm::vec3(0.f));
                    }
                    break;
                }

                case AppState::WALKING_RIGHT:
                case AppState::WALKING_LEFT: {
                    WalkDir& d = currentDir();
                    GLuint tx=0;
                    switch (walkSub) {
                    case WalkSub::IDLE:
                        // Hold first/last frame of the most recently completed sequence.
                        // firstStepTaken disambiguates the genuine pre-walk pose from
                        // a mid-walk transit through stepCount==0 (e.g. after a turn).
                        if      (!firstStepTaken)  tx=d.leftFoot->firstFrame();
                        else if ((stepCount%2)==0) tx=d.rightFoot->lastFrame();
                        else                       tx=d.leftFoot->lastFrame();
                        break;
                    case WalkSub::LEFT_FOOT:  tx=d.leftFoot->current();  break;
                    case WalkSub::RIGHT_FOOT: tx=d.rightFoot->current(); break;
                    case WalkSub::SHAKE_HEAD: tx=d.shakeHead->current(); break;
                    case WalkSub::TURN:       tx=d.turnIn->current();    break;
                    default: break;
                    }
                    if (tx) drawTex(prog,quad,tx,spriteMatrix);
                    else    drawPlaceholder({0.8f,0.5f,0.1f,0.5f},spriteMatrix);  // orange
                    break;
                }
                default: break;
                }

                // Dark green wash, last of all — over the video, over the man,
                // in front of the crowd. Switched on hard at the hand-drawn cut.
                if (greenAlpha > 0.f) {
                    glUseProgram(progPlain);
                    glUniformMatrix4fv(glGetUniformLocation(progPlain,"uModel"),
                                       1,GL_FALSE,glm::value_ptr(glm::mat4(1.f)));
                    glm::vec4 green(GREEN_R,GREEN_G,GREEN_B,greenAlpha);
                    glUniform4fv(glGetUniformLocation(progPlain,"uColor"),
                                 1,glm::value_ptr(green));
                    quad.draw();
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