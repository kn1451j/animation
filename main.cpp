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

#include <mach-o/dyld.h>   // _NSGetExecutablePath, for locating the assets

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
static bool computeAlphaCentroidBottomRGBA(const unsigned char* data, int w, int h,
                                           float& cx, float& by) {
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
    if (sumA <= 0.0) return false;
    cx = (float)(sumAx / sumA);
    by = (float)maxY;
    return true;
}

static bool computeAlphaCentroidBottom(const std::string& pngPath,
                                       float& cx, float& by) {
    int w, h, ch;
    unsigned char* data = stbi_load(pngPath.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!data) return false;
    bool ok = computeAlphaCentroidBottomRGBA(data, w, h, cx, by);
    stbi_image_free(data);
    return ok;
}

// Alpha bounding box of a PNG, in texels. Needed wherever art has to be placed
// by an edge rather than by its centroid — the exit screen stands the man on a
// ground line and centres the exit animation on the sphere.
static bool computeAlphaBBoxRGBA(const unsigned char* data, int w, int h,
                                 float& x0, float& y0, float& x1, float& y1) {
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
    if (maxX < 0) return false;
    x0 = (float)minX; y0 = (float)minY; x1 = (float)maxX; y1 = (float)maxY;
    return true;
}

static bool computeAlphaBBox(const std::string& pngPath,
                             float& x0, float& y0, float& x1, float& y1) {
    int w, h, ch;
    unsigned char* data = stbi_load(pngPath.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!data) return false;
    bool ok = computeAlphaBBoxRGBA(data, w, h, x0, y0, x1, y1);
    stbi_image_free(data);
    return ok;
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
// DOWN out of the walk begins CENTERING: the man (turning right first if he was
// facing left) slides + scales to screen-centre while the background darkens
// smoothly, settling on AUDIENCE_0 — the centred man over a darkened waves
// backdrop with NO crowd yet. Each further DOWN is a "step back" (AUDIENCE_1..3):
// the crowd is progressively revealed (skew + translate + fade), the man drifts
// left and shrinks, and the left darkens. A DOWN at AUDIENCE_3 heads for the exit.
enum class AppState { TITLE, INTRO, BACK_POSE, ENTERING_RIGHT,
                      WALKING_RIGHT, WALKING_LEFT,
                      CENTERING,
                      AUDIENCE_0, AUDIENCE_1, AUDIENCE_2, AUDIENCE_3,
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

// Flat-tint line art that also boils. The man and the three audience levels
// used to ship as 100-frame loops, but those frames were one drawing being
// redrawn — ink sat a p90 of 1 px from frame 0 — so the wobble is generated
// here instead and the assets are single stills (see boil_stills.py).
//
// The noise cell count is a uniform rather than a baked constant: these stills
// are tight crops crossing a range of on-screen sizes, so drawBoiledTinted
// derives it from the rect, keeping cells near the calibrated ~60 screen px
// however big the drawing lands.
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

// Horizontal alpha gradient — uColor is the fill (black here); alpha ramps from
// uLeftA at the screen's left edge to uRightA at the right. Uses VS_SRC so vUV.x
// gives the 0→1 left-to-right position. Drives the audience depth-darkening.
static const char* FS_HGRAD = R"glsl(
#version 330 core
in vec2 vUV; out vec4 frag;
uniform vec3 uColor; uniform float uLeftA; uniform float uRightA;
void main(){ frag = vec4(uColor, mix(uLeftA, uRightA, vUV.x)); })glsl";

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

// drawTinted through the boil shader, for the rect-placed hand-drawn stills.
// rectW/rectH are the drawing's on-screen size in pixels, which is all the
// conversion needs here: a UV offset of d shifts the sampled point by d*rectW
// screen px, so the amplitude is just boilPx over the rect.
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
            "I exist in constant oscillation, disturbed by external forces and pulled towards my unchanging true self...",
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
    // Output pixel format for nextVideoFrame(). The .mp4 backgrounds are opaque
    // and go out as RGB24; the re-encoded sprite sequences carry a real alpha
    // channel and go out as RGBA. Bytes-per-pixel has to track it because
    // sws_scale needs the destination stride.
    AVPixelFormat outFmt = AV_PIX_FMT_RGB24;
    int           outBpp = 3;

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

    bool open(const std::string& path, bool wantAudio=true, bool wantAlpha=false) {
        outFmt = wantAlpha ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24;
        outBpp = wantAlpha ? 4 : 3;
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
            sws=sws_getContext(w,h,vctx->pix_fmt,w,h,outFmt,
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

    // Pull frames until the decoder is genuinely empty, not just until the file
    // runs out of packets. h264 reorders, so it holds several frames inside its
    // pipeline; reading a packet at a time and taking whatever falls out loses
    // however many were still in flight at EOF — it silently truncated the ants
    // sequence from 123 frames to 121. Feeding a null packet puts the decoder in
    // draining mode so the tail comes out too. (Intra-only codecs like qtrle
    // have no delay, which is why only the h264 sequence ever showed it.)
    bool nextVideoFrame(uint8_t* dst) {
        if (!vctx) return false;
        for (;;) {
            int ret=avcodec_receive_frame(vctx,frm);
            if (ret==0) {
                uint8_t* dp[1]={dst}; int ls[1]={w*outBpp};
                sws_scale(sws,frm->data,frm->linesize,0,h,dp,ls);
                return true;
            }
            if (ret!=AVERROR(EAGAIN)) return false;   // AVERROR_EOF or a real error
            if (eof) return false;

            ret=av_read_frame(fmt,pkt);
            if (ret<0) {
                if (loopVideo){ seekToStart(); continue; }
                avcodec_send_packet(vctx,nullptr);    // start draining
                eof=true;
                continue;
            }
            if (pkt->stream_index==vsi) avcodec_send_packet(vctx,pkt);
            av_packet_unref(pkt);
        }
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

    // Wall-clock of the last load()/loadVideo(), so the startup cost can be
    // attributed to a sequence instead of guessed at. Run with ANIM_TIMING=1 to
    // see the per-sequence breakdown; the total always prints.
    long long loadMs = 0;
    static bool timingOn() {
        static const bool on = std::getenv("ANIM_TIMING") != nullptr;
        return on;
    }
    void reportLoad(const std::string& src, const char* kind) const {
        if (timingOn())
            std::cerr<<"[load] "<<src<<"  "<<frames.size()<<' '<<kind
                     <<"  "<<loadMs<<" ms\n";
    }

    bool load(const std::string& dir) {
        const auto t0 = std::chrono::steady_clock::now();
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
        loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now()-t0).count();
        reportLoad(dir, "png");
        return loaded;
    }

    // Decode a re-encoded sequence into the same texture list load() builds.
    // Every frame still lands on the GPU before init() returns — this is not
    // streaming, it is the same eager load with a faster front end. A PNG frame
    // costs ~36 ms through stb_image at -O2; the same frame out of a video
    // stream costs ~1 ms, and that gap is most of the startup wait.
    //
    // wantAlpha=false is for sequences that are opaque everywhere (the alpha
    // channel would be a wasted quarter of the file); the upload still fills
    // alpha with 255 so the shaders sample the same RGBA they always did.
    // Pixels of the first and last decoded frames, kept only when asked for.
    // Two sequences are measured at startup to place them — the exit screen
    // sizes itself off its last frame, the centre sprite off both ends — and
    // those measurements should come from the sequence that actually plays
    // rather than from a PNG directory the video was supposed to replace.
    // Freed by the caller once measured; they are ~8 MB each.
    std::vector<uint8_t> firstFrameRGBA, lastFrameRGBA;

    bool loadVideo(const std::string& path, bool wantAlpha=true,
                   bool keepEnds=false) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!fs::exists(path)) {
            std::cerr<<"[sprite] Video not found: "<<path<<"\n"; return false;
        }
        VideoDecoder dec;
        if (!dec.open(path,/*wantAudio=*/false,wantAlpha)) return false;
        if (dec.vsi<0 || dec.w<=0 || dec.h<=0) {
            std::cerr<<"[sprite] No video stream in "<<path<<"\n";
            dec.close(); return false;
        }
        texW=dec.w; texH=dec.h;

        const size_t px  = (size_t)dec.w*dec.h;
        std::vector<uint8_t> buf(px*dec.outBpp);
        std::vector<uint8_t> rgba;
        if (!wantAlpha) rgba.resize(px*4, 255);

        while (dec.nextVideoFrame(buf.data())) {
            const uint8_t* src=buf.data();
            if (!wantAlpha) {                       // RGB24 -> RGBA, alpha=255
                for (size_t i=0;i<px;++i) {
                    rgba[i*4+0]=buf[i*3+0];
                    rgba[i*4+1]=buf[i*3+1];
                    rgba[i*4+2]=buf[i*3+2];
                }
                src=rgba.data();
            }
            GLuint t; glGenTextures(1,&t);
            glBindTexture(GL_TEXTURE_2D,t);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,dec.w,dec.h,0,
                         GL_RGBA,GL_UNSIGNED_BYTE,src);
            frames.push_back(t);
            if (keepEnds) {
                if (frames.size()==1) firstFrameRGBA.assign(src, src+px*4);
                lastFrameRGBA.assign(src, src+px*4);
            }
        }
        dec.close();
        cur=0; done=false; loaded=!frames.empty();
        loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now()-t0).count();
        reportLoad(path, "vid");
        if (!loaded) std::cerr<<"[sprite] No frames decoded from "<<path<<"\n";
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
    GLuint      progHGrad = 0;   // horizontal alpha gradient — audience darkening
    GLuint      progBoilTint = 0;// hand-drawn line wobble, recoloured — the crowd
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

    // Audience / OUTRO state
    SpriteSeq seqSmoothBug;       // 301 RGBA frames — bug clip, played once in OUTRO
    SpriteSeq seqExit;            // exit/ frame sequence — final exit animation
    bool      pendingCenter = false;   // set when DOWN pressed during a walk
    // The centered man is one digital sprite (enter_right/0089) that replaces
    // the old hand-drawn "standing man" — the whole audience section is drawn
    // with it. manCentroidX/manBottomY are measured off it at load.
    BoilStill manSprite;
    // Waves overlay strength: 0.5 while walking, 2.0 as the audience backdrop.
    // It always draws full-frame now (no camera scale), so no scale member.
    float     wavesAlpha = 0.5f;

    // OUTRO: begins on a key press from AUDIENCE_3 (waves gone). The hand-drawn
    // exit/ animation plays alone, centred on black, with the (quieter) exit
    // voice recording underneath. It plays through once then cuts to black.
    // Runs until the recording finishes (outroEndTime), then quits.
    double     lastOutroFrameTime = 0.0;   // bug-clip cadence timer
    double     lastExitFrameTime  = 0.0;   // exit-anim cadence timer
    double     exitFrameSec       = 1.0/24.0;  // set at exit start; see AUDIENCE_3
    double     outroEndTime       = 0.0;
    static constexpr double OUTRO_FRAME_SEC   = 1.0/24.0;  // bug clip ~24 fps
    static constexpr float  EXIT_AUDIO_GAIN   = 0.5f;      // exit recording volume

    // ── Hand-drawn pull-back (AUDIENCE_0 → AUDIENCE_1/2/3) ───────────────────
    // The audience crowd is one hand-drawn still, pre-split by boil_stills.py
    // into the near figures (l3_back) and the receding rows (l3_front) so the
    // bug clip can be drawn between them. Boiled by FS_BOIL_TINT at 20 fps (they
    // were 100-frame loops that measured as one drawing being redrawn — p90 1 px).
    BoilStill stillAudL3Back, stillAudL3Front;
    static constexpr double HAND_BOIL_HZ = 20.0;   // re-settle rate, was the loop cadence
    static constexpr float  HAND_BOIL_PX = 1.0f;   // p90 line displacement, measured

    // The audience section is a fixed-size framing (no camera zoom): the crowd
    // fills a rect on the right, the man stands on the left. The sense of
    // "stepping back" comes not from scaling the world but, one step per DOWN
    // (indexed by `stage`, eased by stageT), from revealing more of the crowd
    // (fade + recede + deepening skew), drifting the man left and shrinking him,
    // and darkening the left. AUD_WORLD_* place the crowd via drawAudience at the
    // level-3 framing: the bottom-left corner at x=480, the crop spanning 480 →
    // 1920. Earlier levels shrink + offset it off that anchor (AUD_CROWD_*).
    static constexpr float AUD_WORLD_X     = 480.0f;
    static constexpr float AUD_WORLD_SCALE = 1.033000f;
    // The man: a static digital sprite, centroid at MAN_AUD_X[stage], feet on the
    // screen bottom. He lands slightly left of centre (level 0, even before any
    // crowd), then drifts gently further left and shrinks a touch as the crowd is
    // revealed — small, even steps so the moves between levels stay subtle and he
    // ends up fairly large and close to the audience rather than tiny and far off.
    static constexpr float MAN_AUD_X[4]     = { 820.0f, 720.0f, 640.0f, 580.0f };
    static constexpr float MAN_AUD_SCALE[4] = { 0.92f, 0.88f, 0.85f, 0.82f };
    // Per-level scene effects, eased between levels by stageT (see audLerp):
    static constexpr float AUD_SKEW[4]      = { 0.00f, 0.14f, 0.26f, 0.38f }; // left-recede
    static constexpr float AUD_LEFT_DARK[4] = { 0.10f, 0.25f, 0.40f, 0.55f }; // left gradient
    static constexpr float AUD_DARK_BG      = 0.35f;   // flat darken over the video
    // Crowd reveal, per level: hidden at level 0, then faded up while it recedes
    // (scales down) and slides up-and-in from the lower-right, so each step back
    // shows more of the audience. Eased by audLerp like everything else.
    static constexpr float AUD_CROWD_ALPHA[4] = { 0.00f, 0.60f, 0.85f, 1.00f };
    static constexpr float AUD_CROWD_SCALE[4] = { 1.35f, 1.20f, 1.08f, 1.00f };
    static constexpr float AUD_CROWD_DX[4]    = { 260.0f, 150.0f,  60.0f, 0.0f };
    static constexpr float AUD_CROWD_DY[4]    = { 220.0f, 120.0f,  45.0f, 0.0f };
    // Background video zoom, per level: the centering slide pushes the camera IN
    // (bg magnifies to BG_ZOOM[0]); each step back pulls it OUT with the man and
    // crowd, settling at 1.0 (full frame) by the widest level. Scaled about the
    // screen centre in NDC. See bgZoomNow().
    static constexpr float BG_ZOOM[4]         = { 1.50f, 1.32f, 1.16f, 1.00f };
    glm::mat4 audSkew{1.f};   // current left-receding perspective, rebuilt each frame

    // ── Centering transition (walk → AUDIENCE_0) ─────────────────────────────
    // A smooth move into the centred pose: manSprite translates + scales from
    // where the walk left the character to screen-centre (feet on the bottom),
    // while the audience darkening fades in. Settles on AUDIENCE_0 (no crowd
    // yet). The start pose is snapshotted in beginCentering; the end pose is the
    // level-0 man (MAN_AUD_SCALE[0] at MAN_AUD_X[0]) so the handoff is seamless.
    double centerStartTime  = 0.0;
    float  centerStartScale = 1.0f;   // walk sprite draws at native (scale 1)
    float  centerStartCX    = 0.0f;   // where the walk left his centroid X …
    float  centerStartBY    = 0.0f;   // … and his feet (bbox-bottom) Y, on screen
    static constexpr double CENTER_SLIDE_SEC = 1.4;   // slow, deliberate move to centre
    // 0→1 audience-darkening ramp: driven up during CENTERING, held at 1 through
    // the audience so greenAlpha / flat dark / waves come in smoothly, not as a cut.
    float  darkT = 0.f;

    // Once the audience starts, a low-opacity dark green wash goes over
    // everything — video, crowd and man alike — switched on hard as a cut.
    float  greenAlpha = 0.f;
    static constexpr float GREEN_R = 0.05f, GREEN_G = 0.16f, GREEN_B = 0.09f;
    static constexpr float GREEN_MAX = 0.38f;

    int    stage  = 0;      // 0 = AUDIENCE_0 … 3 = AUDIENCE_3
    float  stageT = 1.f;    // 0→1 progress of the move into `stage`
    double stageStartTime = 0.0;
    static constexpr double STAGE_XFADE_SEC = 2.4;   // zoom + cross-fade length (slow step-back)
    float  manCentroidX = 0.f, manBottomY = 0.f;     // manSprite feet anchor

    // The crowd's hand-drawn ink is recoloured to a white outline at draw time
    // so it reads against the darkened backdrop. The man is the digital sprite,
    // drawn as-is.
    glm::vec3 audTint{1.f, 1.f, 1.f};

    // ── Exit screen ──────────────────────────────────────────────────────────
    // The exit animation alone, centred on the screen over plain black. Its
    // drawn extent is measured off the last frame — the only one with the
    // drawing fully in — so the size doesn't depend on the empty margins the
    // 1920×1080 frames carry around it.
    float  exitBoxX0 = 0.f, exitBoxY0 = 0.f, exitBoxX1 = 0.f, exitBoxY1 = 0.f;
    // How much of the screen that extent fills; whichever axis binds first.
    static constexpr float  EXIT_ANIM_H = 0.58f * (float)cfg::WIN_H;
    static constexpr float  EXIT_ANIM_W = 0.70f * (float)cfg::WIN_W;

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

    // Eased per-level value: interpolate arr[stage-1]→arr[stage] by the
    // smoothstepped move progress. At AUDIENCE_0 (stageT=1) it reads arr[0].
    float audLerp(const float arr[4]) const {
        const int   s0 = stage > 0 ? stage - 1 : 0;
        const float t  = stageT * stageT * (3.f - 2.f * stageT);
        return arr[s0] + (arr[stage] - arr[s0]) * t;
    }

    // Current background-video zoom: 1.0 while walking, ramping to BG_ZOOM[0]
    // across the centering slide (eased by darkT), then eased down through the
    // audience levels (BG_ZOOM[stage]) so the camera pulls back with the man and
    // crowd. 1.0 everywhere else.
    float bgZoomNow() const {
        switch (appState) {
            case AppState::CENTERING:
                return 1.0f + (BG_ZOOM[0] - 1.0f) * darkT;
            case AppState::AUDIENCE_0:
            case AppState::AUDIENCE_1:
            case AppState::AUDIENCE_2:
            case AppState::AUDIENCE_3:
                return audLerp(BG_ZOOM);
            default:
                return 1.0f;
        }
    }

    // Background-wave playback speed, tied to the zoom: normal (1.0) while
    // walking and through the centering push-in, then slowing as the camera pulls
    // back across the audience levels (fast/current when zoomed in, slow when
    // zoomed out). Never zero, so it is safe to divide the frame interval by it.
    static constexpr float WAVES_SLOW = 0.40f;   // speed at the fully zoomed-out level
    float wavesSpeedFactor() const {
        switch (appState) {
            case AppState::AUDIENCE_0:
            case AppState::AUDIENCE_1:
            case AppState::AUDIENCE_2:
            case AppState::AUDIENCE_3: {
                // 0 at the widest (zoomed-out) level → 1 at the tightest (in).
                const float f = (bgZoomNow() - 1.0f) / (BG_ZOOM[0] - 1.0f);
                return WAVES_SLOW + (1.0f - WAVES_SLOW) * f;
            }
            default:
                return 1.0f;
        }
    }

    // Left-receding perspective for the audience "zoom out": the bottom-right
    // corner stays fixed (the ground line) while the left edge is pushed inward
    // and foreshortened toward the bottom, so the left reads as farther away.
    // `amt` is the extra projective weight at the far-left edge (0 = flat).
    glm::mat4 sceneSkew(float amt) const {
        if (amt <= 0.f) return glm::mat4(1.f);
        const float W = (float)cfg::WIN_W;
        const float H = (float)cfg::WIN_H;
        // Flip to a right/bottom origin so the weight grows leftward and the
        // bottom edge is the fixed ground line, apply a horizontal perspective,
        // flip back: Hp = T · Hs · T (T is its own inverse).
        float T[3][3]  = {{-1.f,0.f,W},{0.f,-1.f,H},{0.f,0.f,1.f}};
        float Hs[3][3] = {{1.f,0.f,0.f},{0.f,1.f,0.f},{amt/W,0.f,1.f}};
        float tmp[3][3], Hp[3][3];
        mat3Mul(T, Hs, tmp);
        mat3Mul(tmp, T, Hp);
        float Hn[3][3]; homPixToNDC(Hp, Hn);
        return embedHom(Hn);
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
        // Walk_left frames are warped into walk_right space. (The audience man is
        // drawn separately, not through this matrix.)
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

    // Screen rect for manSprite at a given scale, placing his centroid at
    // (centroidX) and his feet (bbox-bottom) at (bottomY). The audience/centering
    // both anchor him this way, so the CENTERING → AUDIENCE_0 handoff is exact.
    glm::mat4 manRectAt(float scale, float centroidX, float bottomY) const {
        return rectMatrix(centroidX - scale*manCentroidX,
                          bottomY   - scale*manBottomY,
                          scale*(float)manSprite.w, scale*(float)manSprite.h);
    }

    // Audience placement: centroid at the given X, feet on the screen bottom.
    glm::mat4 manAudRect(float scale, float centroidX) const {
        return manRectAt(scale, centroidX, (float)cfg::WIN_H);
    }

    // DOWN out of the walk → begin the centering transition. Facing right slides
    // straight in; facing left first turns to the right (reusing the walk-turn
    // machinery), and the slide fires when that turn lands back at IDLE. Returns
    // true if it consumed the frame so the caller breaks out of the walk case.
    bool startCentering(double now) {
        if (appState == AppState::WALKING_RIGHT) { beginCentering(now); return true; }
        // Facing left: kick off a turn to the right, exactly like an opposite-key
        // press, but leave pendingCenter set so the slide fires after the turn.
        if (stepsInCycle == 1) memcpy(Hg_turn, hom::H_left_step_half, sizeof(Hg_turn));
        else                   mat3SetIdentity(Hg_turn);
        appState = AppState::WALKING_RIGHT;
        walkSub  = WalkSub::TURN;
        currentDir().turnIn->reset();
        applyPreTurnAlignment(/*wasRight=*/false);
        updateSpriteMatrix();
        return true;
    }

    // Begin the translate + scale to centre from WALKING_RIGHT IDLE. Snapshots
    // where the walk actually left the character (its idle centroid X and feet Y,
    // on screen), then CENTERING interpolates manSprite from that pose to the
    // level-0 audience pose (screen-centre, feet on the bottom, MAN_AUD_SCALE[0])
    // while the darkening (darkT) fades in.
    void beginCentering(double now) {
        // Snapshot the walk's idle centroid/feet under the live transform
        // (WALKING_RIGHT space here, no left warp). Anchoring the swap to this,
        // not manSprite's own centroid, lands his feet where the walk pose stood.
        float wx, wy;
        idleFrameTexel(/*isLeft=*/false, wx, wy);
        const float pw = Hg_pixel[2][0]*wx + Hg_pixel[2][1]*wy + Hg_pixel[2][2];
        const float px = (Hg_pixel[0][0]*wx + Hg_pixel[0][1]*wy + Hg_pixel[0][2]) / pw;
        const float py = (Hg_pixel[1][0]*wx + Hg_pixel[1][1]*wy + Hg_pixel[1][2]) / pw;
        centerStartCX    = Hg_lift[0][0]*px + Hg_lift[0][1]*py + Hg_lift[0][2];
        centerStartBY    = Hg_lift[1][0]*px + Hg_lift[1][1]*py + Hg_lift[1][2];
        centerStartScale = 1.0f;    // walk frames draw at native full-frame size
        centerStartTime  = now;

        releaseWalkSequences();     // the walk sprites are behind us for good
        walkSub  = WalkSub::IDLE;
        stage = 0;
        stageT = 1.f;
        stageStartTime = now;
        darkT = 0.f;                // ramps up over the slide (no hard cut)
        appState = AppState::CENTERING;
        pendingCenter = false;
        anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
        walkClip.stop();
        // Centred portion plays the title-screen track (scene bg audio hands off).
        titleAudio.audioPos = 0;
        mixer.vidSrc  = &titleAudio;
        mixer.vidSrc2 = nullptr;
        mixer.bgGain.store(1.0f);
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

    // Un-skewed screen rect the crowd fills at the current (eased) reveal level:
    // the level-3 anchor is bottom-left at AUD_WORLD_X, bottom row on the screen
    // bottom, crop spanning 480 → 1920. Earlier levels are larger (closer) and
    // offset down/right (AUD_CROWD_SCALE/DX/DY), so stepping back recedes the
    // crowd into full view. Shared by drawAudience and the bug-bounds helper.
    void crowdRect(float w0, float h0, float& x0, float& y0, float& w, float& h) const {
        const float cScale = AUD_WORLD_SCALE * audLerp(AUD_CROWD_SCALE);
        w = cScale * w0;
        h = cScale * h0;
        x0 = AUD_WORLD_X          + audLerp(AUD_CROWD_DX);
        y0 = (float)cfg::WIN_H - h + audLerp(AUD_CROWD_DY);
    }

    // Draw a crowd layer at the current reveal framing, folding in the current
    // left-receding skew. alpha is the eased per-level crowd opacity.
    void drawAudience(BoilStill& still, float alpha, double now) {
        if (!still.loaded || alpha <= 0.f) return;
        float x0, y0, w, h;
        crowdRect(still.w, still.h, x0, y0, w, h);
        drawBoiledTinted(progBoilTint, quad, still.current(),
                         audSkew * rectMatrix(x0, y0, w, h),
                         w, h, HAND_BOIL_PX,
                         (float)std::floor(now*HAND_BOIL_HZ), audTint, alpha);
    }

    // Fixed (un-skewed) screen rect the crowd fills; used to keep the bug inside.
    // Tracks the same eased reveal framing as drawAudience.
    void audienceRect(float& x0, float& y0, float& x1, float& y1) const {
        const float w0 = stillAudL3Front.w ? stillAudL3Front.w : 1394;
        const float h0 = stillAudL3Front.h ? stillAudL3Front.h : 748;
        float w, h;
        crowdRect(w0, h0, x0, y0, w, h);
        x1 = x0 + w;
        y1 = y0 + h;
    }

    // Hand off to the exit screen: the hand-drawn Exit over black, paced to
    // span the whole voice recording, with everything else released.
    void beginOutro(double now) {
        seqExit.reset();
        // Nothing but the exit drawing is on screen from here, so the crowd,
        // the bug and the man all go.
        manSprite.unload();
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
        const auto initT0 = std::chrono::steady_clock::now();
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
        progHGrad   = makeProgram(VS_SRC,   FS_HGRAD);
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
        if (!seqBg.loadVideo("renders/seq/waves.mov"))
            std::cerr<<"[bg] No frames in renders/seq/waves.mov (run reencode.py)\n";

        // Title-screen ant frames, looped beneath the terminal text. Generated
        // by ants_smooth.py: the isolation pass's wavy top edge, smoothed and
        // feathered, with the original footage filling everything below it.
        // Opaque everywhere, so it ships without an alpha channel: 329 MB of PNGs
        // becomes a 17 MB h264. This is the one sequence that is footage rather
        // than drawing, and lossy is deliberate — see reencode.py.
        if (!seqAnts.loadVideo("renders/seq/ants_smooth.mp4",/*wantAlpha=*/false))
            std::cerr<<"[title] No ant frames in renders/seq/ants_smooth.mp4 (run reencode.py)\n";

        // Walk/turn/pose sprites. All lossless qtrle — they are ~99% empty, so
        // RLE takes them to about a third of the PNGs and a tenth of the decode.
        seqBackPose.load("renders/back_pose");          // single frame, stays a PNG
        seqEnterRight.loadVideo("renders/seq/enter_right.mov");
        seqLeftFoot.loadVideo("renders/seq/walk_right_left_foot.mov");
        seqRightFoot.loadVideo("renders/seq/walk_right_right_foot.mov");
        seqShakeHead.loadVideo("renders/seq/walk_right_shake_head.mov");
        seqTurnLeft.loadVideo("renders/seq/walk_left_turn_left.mov");

        seqWLTurnRight.loadVideo("renders/seq/walk_right_turn_right.mov");
        seqWLLeftFoot.loadVideo("renders/seq/walk_left_left_foot.mov");
        seqWLRightFoot.loadVideo("renders/seq/walk_left_right_foot.mov");
        seqWLShakeHead.loadVideo("renders/seq/walk_left_shake_head.mov");

        dirRight = { &seqLeftFoot,   &seqRightFoot,   &seqShakeHead,   &seqWLTurnRight, +1 };
        dirLeft  = { &seqWLLeftFoot, &seqWLRightFoot, &seqWLShakeHead, &seqTurnLeft,    -1 };

        // The audience man is a single digital sprite — the last frame of the
        // enter-right walk-on, a canonical right-facing standing pose. It is the
        // man for the whole audience section, loaded once and kept. Its anchor
        // (alpha centroid X + bbox-bottom Y) plants it by the feet at any scale.
        static const char* kManSprite = "renders/enter_right/0089.png";
        if (!manSprite.load(kManSprite))
            std::cerr<<"[man] No "<<kManSprite<<"\n";
        if (!computeAlphaCentroidBottom(kManSprite, manCentroidX, manBottomY)) {
            std::cerr<<"[man] Failed to read "<<kManSprite<<" centroid\n";
            manCentroidX = manSprite.w * 0.5f;
            manBottomY   = (float)manSprite.h;
        }
        if (!stillAudL3Back.load("renders/stills/aud_l3_back.png"))
            std::cerr<<"[aud] No renders/stills/aud_l3_back.png (run boil_stills.py)\n";
        if (!stillAudL3Front.load("renders/stills/aud_l3_front.png"))
            std::cerr<<"[aud] No renders/stills/aud_l3_front.png (run boil_stills.py)\n";

        // Bug clip frames — played during AUDIENCE_3. Still the full 1920×1080
        // frame, so BUG_CONTENT_* below stay valid.
        if (!seqSmoothBug.loadVideo("renders/seq/smooth_bug.mov"))
            std::cerr<<"[bug] No frames in renders/seq/smooth_bug.mov (run reencode.py)\n";

        // Exit animation frames — the final outro screen. qtrle is bit-exact, so
        // this is the same pixels the PNGs held, at ~1 ms a frame instead of 36.
        // keepLastFrame because the bbox below is measured off the final frame.
        if (!seqExit.loadVideo("renders/seq/exit.mov",/*wantAlpha=*/true,
                               /*keepEnds=*/true))
            std::cerr<<"[exit] No frames in renders/seq/exit.mov (run reencode.py)\n";

        // The exit animation's bbox, taken from its LAST frame — the only one
        // with the drawing fully in — so it can be sized and centred by what is
        // actually drawn rather than by the frame.
        {
            if (seqExit.lastFrameRGBA.empty() ||
                !computeAlphaBBoxRGBA(seqExit.lastFrameRGBA.data(),
                                      seqExit.texW, seqExit.texH,
                                      exitBoxX0, exitBoxY0, exitBoxX1, exitBoxY1)) {
                std::cerr<<"[exit] Failed to measure exit bbox\n";
                exitBoxX0 = 0.f; exitBoxY0 = 0.f;
                exitBoxX1 = (float)cfg::WIN_W; exitBoxY1 = (float)cfg::WIN_H;
            }
            seqExit.firstFrameRGBA = std::vector<uint8_t>();  // ~8 MB each,
            seqExit.lastFrameRGBA  = std::vector<uint8_t>();  // done with them
        }

        if (!walkClip.load("audios/walking.m4a"))
            std::cerr<<"[audio] walking.m4a not found\n";

        mixer.vidSrc   = &titleAudio;    // looping title music
        mixer.walkPcm  = &walkClip.pcm;
        mixer.walkPos  = &walkClip.pos;
        mixer.walkPlay = &walkClip.playing;
        mixer.walkLoop = &walkClip.looping;
        if (!mixer.init()) std::cerr<<"[audio] mixer init failed\n";

        std::cerr<<"[init] ready in "
                 <<std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now()-initT0).count()
                 <<" ms\n";

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
            bool advBg  = (now-lastBgTime)     >= bgFrameTime / wavesSpeedFactor();
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
                // DOWN ("back") from either direction → defer to the next clean
                // IDLE, then begin the centering transition (startCentering).
                if (downKeyPressed) {
                    pendingCenter = true;
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
                        if (pendingCenter) { startCentering(now); break; }
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
                    if (pendingCenter) { startCentering(now); break; }
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
                // Slide + scale the man to screen-centre while the background
                // darkening fades in, over CENTER_SLIDE_SEC. darkT drives both
                // the man's eased progress (see render) and the darkening. Keys
                // are ignored until it settles onto AUDIENCE_0.
                float t = (float)((now - centerStartTime) / CENTER_SLIDE_SEC);
                if (t > 1.f) t = 1.f;
                darkT      = t*t*(3.f - 2.f*t);            // smoothstep
                greenAlpha = GREEN_MAX * darkT;
                wavesAlpha = 0.5f + (2.0f - 0.5f) * darkT; // 0.5 walking → 2.0 backdrop
                if (t >= 1.f) {
                    darkT = 1.f; greenAlpha = GREEN_MAX; wavesAlpha = 2.0f;
                    stage = 0; stageT = 1.f; stageStartTime = now;
                    anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
                    appState = AppState::AUDIENCE_0;
                }
                break;
            }

            case AppState::AUDIENCE_0:
            case AppState::AUDIENCE_1:
            case AppState::AUDIENCE_2:
            case AppState::AUDIENCE_3: {
                // Ease the move into the current level.
                const bool zooming = (stageT < 1.f);
                if (zooming) {
                    stageT = (float)((now - stageStartTime) / STAGE_XFADE_SEC);
                    if (stageT >= 1.f) {
                        stageT = 1.f;
                        anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
                    }
                }
                // Rebuild the crowd's left-receding skew for this eased depth.
                audSkew = sceneSkew(audLerp(AUD_SKEW));

                // Keys are ignored until the move settles so a fast double-press
                // can't skip a level.
                if (zooming) break;

                if (appState == AppState::AUDIENCE_3) {
                    // Widest level: a bug wanders through the crowd at random.
                    // Roll once a second, only while the last one has finished,
                    // so there is never more than one on screen.
                    if (!bugActive && (now - lastBugRoll) >= BUG_ROLL_SEC) {
                        lastBugRoll = now;
                        if (!seqSmoothBug.frames.empty() && bugRoll(bugRng) < BUG_PROB) {
                            seqSmoothBug.reset();
                            bugActive = true;
                            lastOutroFrameTime = now;
                            // Drop it somewhere in the crowd, inset far enough
                            // that the clip's whole travelled area stays inside.
                            float x0,y0,x1,y1;
                            audienceRect(x0,y0,x1,y1);
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
                    // Only another DOWN heads for the exit.
                    if (downKeyPressed) {
                        anyKeyPressed = rightKeyPressed = leftKeyPressed = downKeyPressed = false;
                        beginOutro(now);
                    }
                } else if (downKeyPressed) {
                    // DOWN = one zoom-out: shrink the man, deepen the skew and the
                    // left darkening (all driven off `stage`/`stageT` at render).
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
                // The exit animation on its own: centred on the screen, scaled
                // up to fill most of it, over plain black. It stops drawing the
                // instant it has played through once — no freeze/hold — while
                // the exit recording keeps playing underneath.
                glClearColor(0,0,0,1);
                glClear(GL_COLOR_BUFFER_BIT);
                glEnable(GL_BLEND);

                if (!seqExit.done) {
                    if (GLuint exTx = seqExit.current()) {
                        // Fit the drawn extent to the screen budget, then place
                        // the frame so that extent's centre is the screen's.
                        const float bw = std::max(1.f, exitBoxX1 - exitBoxX0);
                        const float bh = std::max(1.f, exitBoxY1 - exitBoxY0);
                        const float es = std::min(EXIT_ANIM_W / bw, EXIT_ANIM_H / bh);
                        drawTex(prog, quad, exTx,
                                rectMatrix((float)cfg::WIN_W*0.5f
                                             - es*(exitBoxX0+exitBoxX1)*0.5f,
                                           (float)cfg::WIN_H*0.5f
                                             - es*(exitBoxY0+exitBoxY1)*0.5f,
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
                    // Primary background: looping video (station or grass).
                    // Zoomed in over the centering slide and pulled back out
                    // across the audience levels (scaled about the screen centre).
                    const float bgZoom = bgZoomNow();
                    const glm::mat4 bgM = (bgZoom != 1.0f)
                        ? glm::scale(glm::mat4(1.f), glm::vec3(bgZoom, bgZoom, 1.f))
                        : glm::mat4(1.f);
                    drawTex(prog,quad,bgVidTex,bgM);
                    glEnable(GL_BLEND);

                    const bool inAudience =
                           appState==AppState::AUDIENCE_0
                        || appState==AppState::AUDIENCE_1
                        || appState==AppState::AUDIENCE_2
                        || appState==AppState::AUDIENCE_3;
                    // The centering slide darkens on the way in; the audience holds
                    // it. darkT (0→1) ramps everything during CENTERING and is 1
                    // through the audience, so there is no hard cut.
                    const bool darkScene = inAudience || appState==AppState::CENTERING;

                    // A flat dark layer over the looping video (the "darkened
                    // background"), under the waves, faded in by darkT.
                    if (darkScene && darkT > 0.f) {
                        glUseProgram(progPlain);
                        glUniformMatrix4fv(glGetUniformLocation(progPlain,"uModel"),
                                           1,GL_FALSE,glm::value_ptr(glm::mat4(1.f)));
                        glm::vec4 dark(0.f,0.f,0.f,AUD_DARK_BG*darkT);
                        glUniform4fv(glGetUniformLocation(progPlain,"uColor"),
                                     1,glm::value_ptr(dark));
                        quad.draw();
                    }

                    // Waves overlay — `wavesAlpha` is 0.5 while walking, ramps to
                    // 2.0 across the centering slide, then holds as the audience
                    // backdrop. progOverlay's threshold-amplify shader saturates
                    // wave regions to fully opaque at 2.0 while keeping the PNG's
                    // transparent regions transparent. Shown through the audience,
                    // but NOT the exit (OUTRO) — clean black.
                    bool wavesActive = showWavesOverlay || darkScene;
                    if (wavesActive && bgTex)
                        drawTex(progOverlay,quad,bgTex,glm::mat4(1.f),wavesAlpha);
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

                // ── Centering slide ───────────────────────────────────────────
                // The man translates + scales from where the walk left him to the
                // level-0 audience pose (screen-centre, feet on the bottom), eased
                // by darkT. The end pose is exactly AUDIENCE_0's manAudRect, so the
                // handoff into the audience is seamless. No crowd yet.
                case AppState::CENTERING:
                    if (manSprite.loaded) {
                        const float sc = centerStartScale
                                       + (MAN_AUD_SCALE[0] - centerStartScale)*darkT;
                        const float cx = centerStartCX
                                       + (MAN_AUD_X[0]     - centerStartCX)*darkT;
                        const float by = centerStartBY
                                       + ((float)cfg::WIN_H - centerStartBY)*darkT;
                        drawTex(prog, quad, manSprite.current(), manRectAt(sc, cx, by));
                    }
                    break;

                // ── Audience ─────────────────────────────────────────────────
                // Fixed-size framing: the crowd is revealed level by level (fade +
                // recede, drawAudience folds in the current left-receding skew),
                // the man drifts left and shrinks, and a left darkening gradient
                // deepens with depth. Level 0 shows no crowd (alpha 0).
                case AppState::AUDIENCE_0:
                case AppState::AUDIENCE_1:
                case AppState::AUDIENCE_2:
                case AppState::AUDIENCE_3: {
                    const float crowdA = audLerp(AUD_CROWD_ALPHA);
                    // Crowd (skewed): near figures, the bug between, far rows.
                    drawAudience(stillAudL3Back, crowdA, now);
                    if (bugActive)
                        if (GLuint bugTx = seqSmoothBug.current())
                            drawTex(prog, quad, bugTx,
                                    audSkew * rectMatrix(bugPosX - BUG_SCALE*BUG_CONTENT_CX,
                                                         bugPosY - BUG_SCALE*BUG_CONTENT_CY,
                                                         BUG_SCALE*(float)cfg::WIN_W,
                                                         BUG_SCALE*(float)cfg::WIN_H),
                                    BUG_ALPHA);
                    drawAudience(stillAudL3Front, crowdA, now);

                    // The man: digital sprite, upright (not skewed), drifting left
                    // and shrinking a little more each step back.
                    if (manSprite.loaded)
                        drawTex(prog, quad, manSprite.current(),
                                manAudRect(audLerp(MAN_AUD_SCALE), audLerp(MAN_AUD_X)));

                    // Depth darkening: a black gradient darkest at the LEFT edge
                    // (the far side) and clear at the right, deepening each level.
                    {
                        const float la = audLerp(AUD_LEFT_DARK);
                        if (la > 0.f) {
                            glUseProgram(progHGrad);
                            glUniformMatrix4fv(glGetUniformLocation(progHGrad,"uModel"),
                                               1,GL_FALSE,glm::value_ptr(glm::mat4(1.f)));
                            glUniform3f(glGetUniformLocation(progHGrad,"uColor"),0.f,0.f,0.f);
                            glUniform1f(glGetUniformLocation(progHGrad,"uLeftA"),la);
                            glUniform1f(glGetUniformLocation(progHGrad,"uRightA"),0.f);
                            quad.draw();
                        }
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

// Every asset path in here is relative, so the process has to start in the
// project root. Finder launches apps with the cwd set to "/", and a terminal
// launch inherits wherever you happen to be — so derive the root from the
// executable's own location instead of trusting the cwd.
//
//   Anim.app/Contents/MacOS/anim  →  the folder holding Anim.app
//   ./anim                        →  the folder holding anim
//
// which puts the root in both cases next to audios/, renders/ and vids/.
static void chdirToAssetRoot() {
    char buf[4096];
    uint32_t n = sizeof(buf);
    if (_NSGetExecutablePath(buf, &n) != 0) return;   // path longer than buf
    std::error_code ec;
    fs::path dir = fs::canonical(fs::path(buf).parent_path(), ec);
    if (ec) return;
    if (dir.filename() == "MacOS" && dir.parent_path().filename() == "Contents")
        dir = dir.parent_path().parent_path().parent_path();
    fs::current_path(dir, ec);
    if (ec) std::cerr<<"[init] Could not enter "<<dir<<": "<<ec.message()<<"\n";
}

int main() {
    chdirToAssetRoot();
    App app;
    if (!app.init()) { std::cerr<<"Init failed.\n"; return 1; }
    app.run();
    return 0;
}