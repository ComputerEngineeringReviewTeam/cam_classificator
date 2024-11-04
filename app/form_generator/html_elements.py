"""
Author: Marek Szymański
About: Classes for mimicking and generating HTML code
"""


class Element:
    """
    Class for generating HTML code for single <element>
    """

    def __init__(self, tagname: str, attributes: dict[str, str] = None, content=()):
        """
        Initialize HTML element <tagname>, with attributes and content
        :param tagname: name of the HTML tag
        :param attributes: dictionary of appropriate attributes for the HTML element and their values
        :param content: contents of the element - Elements or other - all will be converted to strings
        """
        self.name = tagname
        self.attributes = attributes
        self.content = content

    def begin_tag(self, show_attributes=True) -> str:
        """
        Returns opening HTML tag for the element, as string
        :param show_attributes: bool controlling whenever to show attributes in the tag or not
        :return: eg. <input type='text' class='big'>
        """
        if show_attributes and self.attributes is not None:
            tag_str = f"<{self.name}"
            for k, v in self.attributes.items():
                tag_str += f" {k}=\"{v}\" "
            return tag_str + ">"
        else:
            return f"<{self.name}>"

    def end_tag(self) -> str:
        """
        Returns ending tag for the element as string
        :return: e.g. "</button>"
        """
        return f"</{self.name}>"

    def full_text(self) -> str:
        """
        Returns full HTML code of the element, including all it's children
        :return: e.g. <div class='red'>
                        <button/>
                      </div>
        """
        if len(self.content) > 0:
            content = ""
            for element in self.content:
                content += f"\n\t{str(element)}"
            return self.begin_tag() + content + "\n" + self.end_tag()
        else:
            return self.begin_tag()

    def __str__(self):
        return self.full_text()

    def short(self) -> str:
        """
        Returns short HTML tag of the element (without attributes)
        :return: "<button>"
        """
        return self.begin_tag(False)


class InputTag(Element):
    """
    Class for creating HTML <input> elements
    TODO: attributes validation
    """

    def __init__(self, attributes: dict = None, content=()):
        super().__init__(tagname='input', attributes=attributes, content=content)


class FormTag(Element):
    """
    Class for creating HTML <form> elements
    TODO: attributes validation
    """

    def __init__(self, attributes: dict = None, content=()):
        super().__init__(tagname='form', attributes=attributes, content=content)
