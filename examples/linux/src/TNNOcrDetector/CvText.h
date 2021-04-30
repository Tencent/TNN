#ifndef CVTEXT_H
#define CVTEXT_H
 
#include <opencv2/opencv.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H
 
class CvText {
public:
 
    /**
           * constructor to initialize a font
           * @param fontName font name
     */
    explicit CvText(const char *fontName);
 
    virtual ~CvText();
 
    /**
           * Set the font properties, keep the default value when the property is empty
           * @param type
           * @param size size
           * @param underline underline
           * @param diaphaneity transparency
     */
    void setFont(int *type, cv::Scalar *size = nullptr,
                 bool *underline = nullptr, float *diaphaneity = nullptr);
 
    /**
           * Restore default font settings
     */
    void restoreFont();
 
    /**
           * Place the content of the text in the specified position (pos) of the frame. The default text color is black. Characters that cannot be output will stop.
           * @param frame output image
           * @param text text content
           * @param pos text position
           * @param color text color
           * @return returns the length of the character that was successfully output, and failed to return -1.
     */
    int putText(cv::Mat &frame, std::string text, cv::Point pos,
                cv::Scalar color = cv::Scalar(0, 0, 0));
 
    /**
             * Place the content of text in the specified position (pos) of the frame. The default color is black. Characters that cannot be output will stop.
             * @param frame output image
             * @param text text content
             * @param pos text position
             * @param color text color
             * @return returns the length of the character that was successfully output, and failed to return -1.
      */
    int putText(cv::Mat &frame, const char *text, cv::Point pos,
                cv::Scalar color = cv::Scalar(0, 0, 0));
 
         //private function area
private:
    /**
           * Output wc to the pos position of the frame
           * @param frame Output Mat
           * @param wc character
           * @param pos position
           * @param color color
     */
    void putWChar(cv::Mat &frame, wchar_t wc, cv::Point &pos, cv::Scalar color);
 
    /**
           * Convert char character array to wchar_t character array
           * @param src char character array
           * @param dst wchar_t character array
           * @param locale locale, mbstowcs function depends on this value to determine the encoding of src
           * @return returns 0 if it succeeds, otherwise -1
     */
    int char2Wchar(const char *&src, wchar_t *&dst, const char *locale = "zh_CN.utf8");
 
         / / private variable area
private:
         FT_Library m_library = NULL; // font
         FT_Face m_face = NULL; // font
 
         // default font output parameters
    int m_fontType;
    cv::Scalar m_fontSize;
    bool m_fontUnderline;
    float m_fontDiaphaneity;
};
 
#endif // CV_TEXT_H